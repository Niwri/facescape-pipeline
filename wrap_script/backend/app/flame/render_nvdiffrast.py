import torch
import numpy as np
import nvdiffrast.torch as dr
import cv2
import imageio
from typing import Optional, Tuple
from tqdm import tqdm
import subprocess
import os
class NvdiffrastMeshRenderer:
    def __init__(self, width: int = 512, height: int = 512, device: str = 'cuda'):
        """
        Initialize the Nvdiffrast mesh renderer.
        Args:
            width: Output image width
            height: Output image height
            device: Device to use ('cuda' or 'cpu', but 'cuda' is highly recommended)
        """
        self.width = width
        self.height = height
        self.device = device
        # Initialize the rasterizer context
        self.ctx = dr.RasterizeCudaContext()
    def setup_camera(self, distance: float = 3.0, elevation: float = 0.0, azimuth: float = 0.0):
        """
        Setup camera parameters for rendering.
        Args:
            distance: Distance from object
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
        """
        # Convert angles to radians
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        # Camera position in spherical coordinates
        x = distance * np.cos(elev_rad) * np.sin(azim_rad)
        y = distance * np.sin(elev_rad)
        z = distance * np.cos(elev_rad) * np.cos(azim_rad)
        # Create view matrix (looking at origin)
        eye = torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
        at = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        # Compute view matrix
        self.view_matrix = self._look_at(eye, at, up)
        # Projection matrix (perspective)
        fov = 45.0
        aspect = self.width / self.height
        near, far = 0.1, 100.0
        self.proj_matrix = self._perspective(fov, aspect, near, far)
        # Combined MVP matrix
        self.mvp = torch.matmul(self.proj_matrix, self.view_matrix)

    def setup_camera_from_extrinsics(self, intrinsics: np.ndarray, extrinsic: np.ndarray):
        """
        intrinsics: (3,3)
        extrinsic: (4,4) world-to-camera. Will need to fix
        """ 
        # Convert to torch
        K = torch.tensor(intrinsics, dtype=torch.float32, device=self.device)
        M_wc = torch.tensor(extrinsic, dtype=torch.float32, device=self.device)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        w = self.width
        h = self.height

        far = 100.0
        near = 0.01

        # --- Build OpenGL projection matrix ---
        proj = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        proj[0, 0] = 2 * fx / w
        proj[1, 1] = 2 * fy / h
        proj[0, 2] = 1 - 2 * cx / w
        proj[1, 2] = 2 * cy / h - 1
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1

        # --- Convert world→camera to camera→world (OpenGL view matrix) ---

        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=self.device))
        view = flip @ torch.linalg.inv(M_wc)

        # view = flip @ M_wc

        self.proj_matrix = proj
        self.view_matrix = view
        self.mvp = proj @ view

    def _look_at(self, eye, at, up):
        """Create a look-at view matrix."""
        f = torch.nn.functional.normalize(at - eye, dim=0)
        s = torch.nn.functional.normalize(torch.cross(f, up), dim=0)
        u = torch.cross(s, f)
        view = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[3, 3] = 1.0
        view[0, 3] = -torch.dot(s, eye)
        view[1, 3] = -torch.dot(u, eye)
        view[2, 3] = torch.dot(f, eye)
        return view
    def _perspective(self, fov, aspect, near, far):
        """Create a perspective projection matrix."""
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        proj = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj
    def render_frame(self, vertices: np.ndarray, faces: np.ndarray,
                 vertex_colors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render a single frame using CUDA backend.
        """
        # Ensure correct dtypes
        vertices_torch = torch.from_numpy(vertices.astype(np.float32)).to(self.device)
        faces_torch_int32 = torch.from_numpy(faces.astype(np.int32)).to(self.device)   # 🔑 for rasterizer
        faces_torch_long  = faces_torch_int32.long()  # 🔑 for scatter_add (lighting)

        vertices_torch = vertices_torch.unsqueeze(0)  # (1, V, 3)
        vertices_homo = torch.cat([
            vertices_torch,
            torch.ones(1, vertices_torch.shape[1], 1, device=self.device)
        ], dim=2)  # (1, V, 4)
        vertices_clip = torch.matmul(vertices_homo, self.mvp.T)

        # Rasterize → requires int32 faces
        rast, _ = dr.rasterize(self.ctx, vertices_clip, faces_torch_int32,
                            resolution=[self.height, self.width])

        if vertex_colors is not None:
            colors_torch = torch.from_numpy(vertex_colors.astype(np.float32)).to(self.device)
            colors_torch = colors_torch.unsqueeze(0)
        else:
            # Compute normals → requires int64 faces
            colors_torch = self._compute_vertex_lighting(vertices_torch, faces_torch_long)

        color_image, _ = dr.interpolate(colors_torch, rast, faces_torch_int32)
        color_image = dr.antialias(color_image, rast, vertices_clip, faces_torch_int32)

        image = color_image[0].cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image

    
    
    def _compute_vertex_lighting(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute simple vertex lighting based on normals.
        Args:
            vertices: Vertex positions (1, V, 3)
            faces: Face indices (F, 3)
        Returns:
            Vertex colors (1, V, 3)
        """
        # Compute face normals
        v0 = vertices[0, faces[:, 0]]  # (F, 3)
        v1 = vertices[0, faces[:, 1]]  # (F, 3)
        v2 = vertices[0, faces[:, 2]]  # (F, 3)
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # Accumulate face normals to vertices
        vertex_normals = torch.zeros_like(vertices[0])  # (V, 3)
        vertex_counts = torch.zeros(vertices.shape[1], device=self.device)
        for i in range(3):
            vertex_normals.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), face_normals)
            vertex_counts.scatter_add_(0, faces[:, i], torch.ones_like(faces[:, i], dtype=torch.float32))
        # Average normals
        vertex_normals = vertex_normals / (vertex_counts.unsqueeze(1) + 1e-8)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
        # Simple lighting (light from above and front)
        light_dir = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        light_dir = torch.nn.functional.normalize(light_dir, dim=0)
        # Lambertian shading
        lambertian = torch.clamp(torch.sum(vertex_normals * light_dir, dim=1), 0.2, 1.0)
        # Convert to RGB (grayish color)
        base_color = torch.tensor([0.8, 0.8, 0.9], device=self.device)
        colors = lambertian.unsqueeze(1) * base_color.unsqueeze(0)
        return colors.unsqueeze(0)  # Add batch dimension

def render_mesh_animation(vertices_sequence: np.ndarray, faces: np.ndarray,
                         output_path: str = "animation.mp4", fps: int = 25,
                         width: int = 512, height: int = 512,
                         camera_distance: float = 3.0,
                         rotate_camera: bool = False,
                         vertex_colors: Optional[np.ndarray] = None,
                         audio_path: Optional[str] = None) -> None:
    """
    Render a mesh animation sequence to video using Nvdiffrast, and mux with audio (via FFmpeg).
    """
    renderer = NvdiffrastMeshRenderer(width=width, height=height)
    tmp_video_path = output_path.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(tmp_video_path, fourcc, fps, (width, height))
    num_frames = vertices_sequence.shape[0]
    try:
        for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
            if rotate_camera:
                azimuth = (frame_idx / num_frames) * 360.0
            else:
                azimuth = 0.0
            renderer.setup_camera(distance=camera_distance, elevation=0.0, azimuth=azimuth)
            current_vertices = vertices_sequence[frame_idx]
            image = renderer.render_frame(current_vertices, faces, vertex_colors)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = cv2.flip(image_bgr, 0)
            video_writer.write(image_bgr)
    finally:
        video_writer.release()
    print(f"Video (no audio) saved to: {tmp_video_path}")
    if audio_path is not None:
        print("Muxing audio with ffmpeg...")
        # ffmpeg command:
        # -i video -i audio
        # -c:v copy   (don’t reencode video)
        # -c:a aac    (encode audio as AAC for MP4)
        # -shortest   (cut to shortest stream: trims audio if longer; cuts video if shorter)
        # -af apad    (pads audio with silence if shorter)
        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i", tmp_video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-af", "apad",
            output_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(tmp_video_path)
    else:
        os.rename(tmp_video_path, output_path)
    print(f"Animation saved to: {output_path}")
def render_flame_animation(Vs: np.ndarray, faces: np.ndarray,
                           output_path: str = "flame_animation.mp4",
                           fps: int = 25,
                           audio_path: Optional[str] = None) -> None:
    """
    Render FLAME mesh animation and mux with optional audio.
    """
    render_mesh_animation(
        vertices_sequence=Vs,
        faces=faces,
        output_path=output_path,
        fps=fps,
        width=512,
        height=512,
        camera_distance=0.6,
        rotate_camera=False,
        audio_path=audio_path
    )

def render_faceview_animation(vertices_sequence: np.ndarray, faces: np.ndarray,
                         intrinsic: np.ndarray,
                         extrinsic: np.ndarray,
                         output_path: str = "animation.mp4", fps: int = 25,
                         width: int = 512, height: int = 512,
                         vertex_colors: Optional[np.ndarray] = None,
                         audio_path: Optional[str] = None) -> None:
    """
    Render a mesh animation sequence to video using Nvdiffrast, and mux with audio (via FFmpeg).
    """
    renderer = NvdiffrastMeshRenderer(width=width, height=height)
    tmp_video_path = output_path.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(tmp_video_path, fourcc, fps, (width, height))
    num_frames = vertices_sequence.shape[0]
    try:
        for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
            renderer.setup_camera_from_extrinsics(intrinsic, extrinsic)
            current_vertices = vertices_sequence[frame_idx]
            image = renderer.render_frame(current_vertices, faces, vertex_colors)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = cv2.flip(image_bgr, 0)
            video_writer.write(image_bgr)
    finally:
        video_writer.release()
    print(f"Video (no audio) saved to: {tmp_video_path}")
    if audio_path is not None:
        print("Muxing audio with ffmpeg...")
        # ffmpeg command:
        # -i video -i audio
        # -c:v copy   (don’t reencode video)
        # -c:a aac    (encode audio as AAC for MP4)
        # -shortest   (cut to shortest stream: trims audio if longer; cuts video if shorter)
        # -af apad    (pads audio with silence if shorter)
        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i", tmp_video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-af", "apad",
            output_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(tmp_video_path)
    else:
        os.rename(tmp_video_path, output_path)
    print(f"Animation saved to: {output_path}")