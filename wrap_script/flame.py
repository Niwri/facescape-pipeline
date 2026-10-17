from time import perf_counter
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import imageio
import cv2
import logging

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    AmbientLights,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    OrthographicCameras,
    TexturesVertex,
    PerspectiveCameras,
    BlendParams,
    HardFlatShader
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.ops import interpolate_face_attributes

class VertexColorShader(ShaderBase):
    """
    A shader that simply returns interpolated vertex colors without any lighting.
    This preserves the exact vertex values (used for encoding 3D coordinates as RGB).
    """
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__(device=device, cameras=cameras, blend_params=blend_params)
    
    def forward(self, fragments, meshes, **kwargs):
        # Get blend params
        blend_params = kwargs.get("blend_params", self.blend_params)
        if blend_params is None:
            blend_params = BlendParams()
        
        # Sample textures at fragment locations using barycentric interpolation
        # This gives us the interpolated vertex colors
        texels = meshes.sample_textures(fragments)  # (N, H, W, K, C)
        
        # Use hard blending (take the nearest fragment's color)
        images = hard_rgb_blend(texels, fragments, blend_params)
        
        return images


class SafeSoftPhongShader(ShaderBase):
    """
    Soft Phong-like shader which avoids cameras.get_camera_center() inverse.
    Camera center is derived directly from camera R/T.
    """

    def _camera_center_from_rt(self, cameras: PerspectiveCameras) -> torch.Tensor:
        if hasattr(cameras, "R") and hasattr(cameras, "T") and cameras.R is not None and cameras.T is not None:
            R = cameras.R
            T = cameras.T
            # World->view: X_cam = X_world @ R + T
            # Camera center in world: C = -T @ R^T
            return -torch.bmm(T.unsqueeze(1), R.transpose(1, 2)).squeeze(1)
        # Fallback to default behavior if R/T is unavailable.
        return cameras.get_camera_center()

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            raise ValueError("Cameras must be specified for SafeSoftPhongShader.")
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        texels = meshes.sample_textures(fragments)

        verts = meshes.verts_packed()  # (V, 3)
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_verts = verts[faces]
        faces_normals = vertex_normals[faces]

        pixel_coords = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        )
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        camera_center = self._camera_center_from_rt(cameras)
        light_diffuse = lights.diffuse(normals=pixel_normals, points=pixel_coords)
        light_specular = lights.specular(
            normals=pixel_normals,
            points=pixel_coords,
            camera_position=camera_center,
            shininess=materials.shininess,
        )
        ambient = materials.ambient_color * lights.ambient_color
        diffuse = materials.diffuse_color * light_diffuse
        specular = materials.specular_color * light_specular
        if ambient.ndim != diffuse.ndim:
            ambient = ambient[:, None, None, None, :]

        colors = (ambient + diffuse) * texels + specular
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(colors, fragments, blend_params, znear=znear, zfar=zfar)
        return images

import numpy as np

UV_MIN = torch.tensor([[[-0.032555654644966125, -0.06085548922419548, -0.054335832595825195]]])
UV_MAX = torch.tensor([[[0.03436008468270302, 0.0675620436668396, 0.03944191709160805]]])

def norm_c(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor of shape [T, V, 3] to [-1, 1] per channel.

    Args:
        x (torch.Tensor): Input tensor with shape [T, V, 3].

    Returns:
        torch.Tensor: Normalized tensor with same shape, values in [-1, 1].
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("Input must have shape [T, V, 3]")
    
    min_vals = UV_MIN.to(x.device)
    max_vals = UV_MAX.to(x.device)
    # Avoid division by zero
    denom = (max_vals - min_vals).clamp(min=1e-8)

    # Normalize [0,1], then scale [-1,1]
    x01 = (x - min_vals) / denom
    x_norm = x01 * 2 - 1
    return x_norm, min_vals, max_vals

def denorm_c(x_norm: torch.Tensor, min_vals: torch.Tensor, max_vals: torch.Tensor):
    """
    Invert the normalization: [-1,1] → original range.
    """
    denom = (max_vals - min_vals).clamp(min=1e-8)
    x01 = (x_norm + 1) / 2
    x = x01 * denom + min_vals
    return x

def denorm_c(x_norm: torch.Tensor):
    """
    Invert the normalization: [-1,1] → original range.
    """
    # min_vals, max_vals = torch.tensor([[[-0.007567240856587887, -0.02616330422461033, -0.022287581115961075]]], device=x_norm.device), torch.tensor([[[0.007499536499381065, 0.009242897853255272, 0.00502586318179965]]], device=x_norm.device)
    min_vals = UV_MIN.to(x_norm.device)
    max_vals = UV_MAX.to(x_norm.device)
    denom = (max_vals - min_vals).clamp(min=1e-8)
    x01 = (x_norm + 1) / 2
    x = x01 * denom + min_vals
    return x

def batch_delta_uv(head, path_list, sample_frames=-1, resolution=256, rotation=False, norm=False, orig=False):
    out = []
    orig = []
    min_frame = 1e8

    for path in path_list:
        head.loadSequence(path)

        identity_pose = head.LSB(rotation=False, identity=True)
        seq = head.LSB(rotation=False)
        # seq[..., 1] = seq[..., 1] * -1
        if sample_frames == -1: sample_frames = seq.shape[0]

        first = seq[0:1, ...] - identity_pose[0:1, ...]     # Get first difference from identity to sequence
        remainder = seq[1:, ...] - seq[:-1, ...]
        delta = torch.cat([first, remainder], dim=0).cumsum(dim=0)
        normed, _, _ = norm_c(delta)
        uvMesh = head.convertUV(customSeq=normed, rotation=False, norm=norm)
        
        frame, _, _ = normed.shape    
        if frame < min_frame: min_frame = frame                                       
        uv = head.get_uv_animation(uvMesh, resolution=resolution, sample_frames=sample_frames) 
        out.append(uv[..., :3].unsqueeze(0))
        orig.append(seq[:sample_frames])

    out = [item[:, :min_frame, ...] for item in out]
    
    if orig:
        return torch.concat(out, dim=0), torch.concat(orig, dim=0)
    else:
        return torch.concat(out, dim=0)
    

def proj_v(
    verts_3d: torch.Tensor, 
) -> torch.Tensor:

    verts_2d = torch.stack([
        verts_3d[..., 0] / verts_3d[..., 2],  # X/Z
        verts_3d[..., 1] / verts_3d[..., 2],  # Y/Z
        verts_3d[..., 2],
    ], dim=-1)

    return verts_2d

def unproj_v(
    verts_3d: torch.Tensor, 
) -> torch.Tensor:

    verts_2d = torch.stack([
        verts_3d[..., 0] * verts_3d[..., 2],  # X/Z
        verts_3d[..., 1] * verts_3d[..., 2],  # Y/Z
        verts_3d[..., 2],
    ], dim=-1)

    return verts_2d

def apply_extri(
    verts_3d: torch.Tensor, 
    extri: torch.Tensor,
):
    verts_3d_cam = einops.einsum(
        extri[:, :3, :3], 
        verts_3d, 
        "N_t i j, N_t V j -> N_t V i"
    )
    verts_3d_cam = verts_3d_cam + extri[:, None, :3, 3]
    
    return verts_3d_cam

def proj(
    verts_3d_cam: torch.Tensor, 
):
    verts_2d = torch.stack([
        verts_3d_cam[..., 0] / verts_3d_cam[..., 2],  # X/Z
        verts_3d_cam[..., 1] / verts_3d_cam[..., 2],  # Y/Z
        verts_3d_cam[..., 2],
    ], dim=-1)
    return verts_2d

def apply_intri(
    verts_2d: torch.Tensor,
    intri: torch.Tensor,
):
    fx = intri[:, 0, 0][:, None]
    fy = intri[:, 1, 1][:, None]
    cx = intri[:, 0, 2][:, None]
    cy = intri[:, 1, 2][:, None]

    img_2d = torch.stack([
        verts_2d[..., 0] * fx + cx,
        verts_2d[..., 1] * fy + cy,
        verts_2d[..., 2],
    ], dim=-1)

    return img_2d

def proj_intri(
    verts_3d_cam: torch.Tensor, 
    intri: torch.Tensor,
):
    fx = intri[:, 0, 0][:, None]
    fy = intri[:, 1, 1][:, None]
    cx = intri[:, 0, 2][:, None]
    cy = intri[:, 1, 2][:, None]

    # Pixel coordinates
    verts_2d = torch.stack([
        verts_3d_cam[..., 0] / verts_3d_cam[..., 2] * fx + cx,
        verts_3d_cam[..., 1] / verts_3d_cam[..., 2] * fy + cy,
        verts_3d_cam[..., 2] / verts_3d_cam[..., 2].mean(dim=-1, keepdim=True) * (fx + fy) / 2,
    ], dim=-1)

    return verts_2d

def project_vertices_paired(
    verts_3d: torch.Tensor, 
    extri: torch.Tensor,
    intri: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    # ... same camera transformation ...
    
    verts_3d_cam = einops.einsum(
        extri[:, :3, :3], 
        verts_3d, 
        "N_t i j, N_t V j -> N_t V i"
    )
    verts_3d_cam = verts_3d_cam + extri[:, None, :3, 3]

    fx = intri[:, 0, 0][:, None]
    fy = intri[:, 1, 1][:, None]
    cx = intri[:, 0, 2][:, None]
    cy = intri[:, 1, 2][:, None]

    if normalize:
        # Normalized image coordinates: just X/Z and Y/Z
        verts_2d = torch.stack([
            verts_3d_cam[..., 0] / verts_3d_cam[..., 2],  # X/Z
            verts_3d_cam[..., 1] / verts_3d_cam[..., 2],  # Y/Z
            verts_3d_cam[..., 2],
        ], dim=-1)
    else:
        # Pixel coordinates
        verts_2d = torch.stack([
            verts_3d_cam[..., 0] / verts_3d_cam[..., 2] * fx + cx,
            verts_3d_cam[..., 1] / verts_3d_cam[..., 2] * fy + cy,
            verts_3d_cam[..., 2] / verts_3d_cam[..., 2].mean(dim=-1, keepdim=True) * (fx + fy) / 2,
        ], dim=-1)

    return verts_2d


def dot(
    x: torch.Tensor,
    y: torch.Tensor,
    dim=-1,
    keepdim=False,
) -> torch.Tensor:
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def safe_length(
    x: torch.Tensor,
    dim=-1,
    keepdim=False,
    eps=1e-20,
) -> torch.Tensor:
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x, dim=dim, keepdim=keepdim), min=eps))

def batch_rodrigues(
    rot_vecs: torch.Tensor, epsilon=1e-8  # (B, 3)
) -> torch.Tensor:  # (B, 3, 3)
    """Calculates the rotation matrices for a batch of rotation vectors.
    Reference:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    Parameters
    ----------
    rot_vecs: torch.tensor Bx3
        array of B axis-angle vectors
    Returns
    -------
    R: torch.tensor Bx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = safe_length(rot_vecs, keepdim=True, eps=epsilon)  # (B, 1)
    rot_dir = rot_vecs / angle  # (B, 3)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)  # each (B, 1)

    zeros = torch.zeros(
        (batch_size, 1), dtype=torch.float32, device=device
    )  # (B, 1)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    )  # (B, 9)
    K = K.view(batch_size, 3, 3)  # (B, 3, 3)

    ident: torch.Tensor = torch.eye(
        3, dtype=torch.float32, device=device
    ).unsqueeze(
        dim=0
    )  # (1, 3, 3)
    cos = torch.unsqueeze(torch.cos(angle), dim=1)  # (B, 1, 1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)  # (B, 1, 1)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_vertices(
    transform: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """
    Transforms a list of vertices by a transform.

    Parameters
    ----------
    transform: torch.Tensor [B, 4, 4]
    vertices: torch.Tensor [B, N, 3]

    Returns
    -------
    transformed_vertices: torch.Tensor [B, N, 3]
    """
    transformed_verts = transform[:, :3, :3] @ vertices.permute(0, 2, 1)
    transformed_verts = transformed_verts + transform[:, :3, [3]]
    return transformed_verts.permute(0, 2, 1)

def draw_vertices_on_image(
    img: np.ndarray,
    vertices: np.ndarray,
    vert_colors: np.ndarray = None,
):
    if vert_colors is None:
        vert_colors = np.array([[20, 255, 20]], dtype=np.uint8).repeat(vertices.shape[0], axis=0)

    verts_inside_frame = np.logical_and(
        np.logical_and(vertices[:, 0] > 1, vertices[:, 0] < img.shape[1] - 1),
        np.logical_and(vertices[:, 1] > 1, vertices[:, 1] < img.shape[0] - 1),
    )
    vertices = vertices[verts_inside_frame, :]
    vert_colors = vert_colors[verts_inside_frame, :]

    x_indices = np.floor(vertices[:, 0]).astype(int)
    y_indices = np.floor(vertices[:, 1]).astype(int)

    img[y_indices, x_indices,  :] = vert_colors
    img[y_indices + 1, x_indices,  :] = vert_colors
    img[y_indices, x_indices + 1,  :] = vert_colors
    img[y_indices - 1, x_indices,  :] = vert_colors
    img[y_indices, x_indices - 1,  :] = vert_colors

    return img

def draw_lines_on_image(
    image: np.ndarray, 
    verts: np.ndarray, 
    lines: np.ndarray, 
    vert_colors: np.ndarray = None, 
    thickness: float = 1,
):
    for line in lines:
        x1 = int(verts[line[0], 0])
        y1 = int(verts[line[0], 1])
        x2 = int(verts[line[1], 0])
        y2 = int(verts[line[1], 1])
        color = (20, 255, 20)
        if vert_colors is not None:
            color = ((vert_colors[line[0]] // 2 + vert_colors[line[1]] // 2)).tolist()
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

def is_invalid(t):
    return t is None or not torch.is_tensor(t) or torch.isnan(t).any() or t.numel() != 2

OPENCV2PYTORCH3D = torch.eye(4)
OPENCV2PYTORCH3D[1, 1] = -1
OPENCV2PYTORCH3D[2, 2] = -1
UV_PATH = "flame_uv_flowface.obj"

class Flame(nn.Module): # Must inherit from nn.Module to use register_buffer
    def __init__(
        self, 
        flamePath : str = "flame_model/FLAME2020/flame2023_no_jaw.pkl",
        device : str = "cuda",
        n_shape_params: int = 150,
        n_expr_params: int = 65,
        uvPath :str = UV_PATH
    ):
        super(Flame, self).__init__() # Initialize the nn.Module

        obj = self._load_flame_model(flamePath)
        self.device = device
        
        # Identity Faces - Registered as Buffers
        self.register_buffer('verts', torch.tensor(obj['v_template']).float())
        self.register_buffer('faces', torch.tensor(obj['f']).long())

        # PCs on shape, expression, and pose - Registered as Buffers
        self.register_buffer('shapeEigen', torch.tensor(obj['shapedirs'][..., :300]).float())
        self.register_buffer('poseEigen', torch.tensor(obj['posedirs']).float())
        self.register_buffer('exprEigen', torch.tensor(obj["shapedirs"][..., 300:700]).float())

        # Parameters for LSB - Registered as Buffers
        self.register_buffer('skinningWt', torch.tensor(obj['weights']).float())
        self.register_buffer('jointRegressor', torch.tensor(obj['J_regressor']).float())
        self.register_buffer('joint_parents', torch.tensor(obj['kintree_table']).long())
        
        # UV info
        verts_raw, faces_raw, aux = load_obj(uvPath)

        uv_coords = aux.verts_uvs

        self.facesUV3d, self.vertsUV3d = faces_raw.verts_idx.to(self.device), verts_raw.to(self.device)
        # uv_coords = uv_coords * 2 - 1
        # uv_coords[..., 1] = -uv_coords[..., 1]
        # Registering UV and 3D topology as Buffers
        self.register_buffer('facesUV', faces_raw.textures_idx)
        self.register_buffer('vertsUV', uv_coords, persistent=False)
        self.register_buffer('faces3d', faces_raw.verts_idx)
        self.register_buffer('verts3d', verts_raw)

        # Standard attributes (Non-Tensors)
        self.nFaces = self.faces3d.shape[0]
        self.seq = None 

        # Move the entire module (and all registered buffers) to the device
        self.to(self.device)

    @staticmethod
    def _load_flame_model(path: str) -> dict:
        """Load a FLAME model from .npz or .pkl, returning a dict of numpy arrays.

        For .pkl files the loader avoids importing chumpy (which is broken on
        Python 3.11+ / NumPy 2.x) by temporarily injecting shim modules into
        ``sys.modules`` so that pickle never touches the real chumpy package.
        """
        if path.endswith(".pkl"):
            import numpy as _np
            import inspect as _inspect

            # Pre-import numpy.ma and scipy.sparse so their module-level
            # initialisation runs with the real numpy scalar types BEFORE
            # we temporarily patch np.bool/np.int/… below.  Without this,
            # pickle.load can trigger a fresh scipy import which in turn
            # does ``from numpy import *`` → imports numpy.ma.core → its
            # MaskedConstant() uses patched ``bool`` (no .view()) and crashes.
            try:
                import numpy.ma.core  # noqa: F401
            except Exception:
                pass
            try:
                import scipy.sparse  # noqa: F401
            except Exception:
                pass

            _saved_np = {}
            for attr in ("bool", "int", "float", "complex", "object", "unicode", "str"):
                if hasattr(_np, attr):
                    _saved_np[attr] = getattr(_np, attr)
            _np.bool = bool
            _np.int = int
            _np.float = float
            _np.complex = complex
            _np.object = object
            _np.unicode = str
            _np.str = str

            _had_getargspec = hasattr(_inspect, "getargspec")
            if not _had_getargspec:
                _inspect.getargspec = _inspect.getfullargspec

            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(path, "rb") as f:
                        raw = pickle.load(f, encoding="latin1")
            finally:
                for attr, val in _saved_np.items():
                    setattr(_np, attr, val)
                if not _had_getargspec:
                    del _inspect.getargspec

            obj = {}
            for key, value in raw.items():
                if not hasattr(value, "shape"):
                    continue
                arr = value
                if callable(getattr(arr, "todense", None)):
                    arr = arr.todense()
                if key == "f":
                    obj[key] = np.array(arr, dtype=np.int32).squeeze()
                elif np.issubdtype(np.asarray(arr).dtype, np.floating):
                    obj[key] = np.array(arr, dtype=np.float32).squeeze()
                else:
                    obj[key] = np.array(arr, dtype=np.int64).squeeze()
            if "kintree_table" in obj:
                obj["kintree_table"][0, 0] = -1
            return obj
        return np.load(path)

    def loadSequence(
        self,
        sequencePath : str
    ):
        seq = np.load(sequencePath)
        
        if isinstance(seq["fx"], np.ndarray):
            seq = dict(seq)
            seq["fx"] = seq["fx"][0]
            seq["fy"] = seq["fy"][0]
            seq["cx"] = seq["cx"][0]
            seq["cy"] = seq["cy"][0]

        # Blend shape parameter (1, 150), (nFrames, 65)
        self.beta, self.psi = torch.tensor(seq['shape']).to(self.device), torch.tensor(seq['expr']).to(self.device)
        self.nFrames = self.psi.shape[0]

        # Head Positions
        self.extr = torch.tensor(seq['extr']).to(self.device)
        self.rot, self.tra = torch.tensor(seq['rot']).to(self.device), -torch.tensor(seq['tra']).to(self.device)
        self._rot = self.rot.clone()
        self.rot[..., 0] = (self.rot[..., 0] - torch.pi) 
        # self.rot[..., 1] += 12 * (torch.pi / 180)
        self.rot[..., 1] = 1 * self._rot[..., 2] 
        self.rot[..., 2] = -1 * self._rot[..., 1] 
        # self.rot[..., 1] *= 0 # self.rot[..., 1] *= 1
        # self.rot[..., 2] *= 0 # self.rot[..., 1] *= 1
        self.tra[..., 0] *= -1

        # Camera parameters
        self.focalLength = torch.ones(self.nFrames, 2).to(self.device)
        self.principalPoint = torch.ones(self.nFrames, 2).to(self.device)

        self.focalLength[:, 0] = torch.tensor(seq['fx']).to(self.device).squeeze() 
        self.focalLength[:, 1] = torch.tensor(seq['fy']).to(self.device).squeeze()

        self.principalPoint[:, 0] = torch.tensor(seq['cx']).to(self.device).squeeze() 
        self.principalPoint[:, 1] = torch.tensor(seq['cy']).to(self.device).squeeze()

        self.extrRot = self.extr[:, :3, :3]
        self.extrTra = self.extr[:, :3, 3]

    def blendShape(self):
        # Get Number of basis used

        if len(self.beta.shape) == 2 and torch.all(self.beta[0] == self.beta[1]):
            self.beta = self.beta[0]

        nPC = self.beta.shape[0]
        self.shapeEigen = self.shapeEigen[..., :nPC]

        # Expand beta
        if self.beta.shape[0] != self.nFrames:
            self.beta = self.beta.unsqueeze(0).expand(self.nFrames, nPC) 

        return einops.einsum(
            self.beta, 
            self.shapeEigen,
            "b betas, V xyz betas -> b V xyz"
        )

        return torch.einsum('b i v, b n v -> b n i ', self.beta, U).squeeze(-1)

    def blendExpr(self):
        # Get number of basis
        nPC = self.psi.shape[1]

        assert self.psi.shape[1] == 65

        return einops.einsum(
            self.psi, 
            self.exprEigen[:, :, :nPC],
            "b betas, V xyz betas -> b V xyz"
        )

    def identityRotation(
        self,
        perFrameVerts
    ):
        # initialize identity rotation for 5 joints across all frames
        rotation = torch.eye(3, device=self.device).unsqueeze(0).expand(5, 3, 3).unsqueeze(0).expand(self.nFrames, 5, 3, 3)

        _U = einops.rearrange(self.poseEigen, "N xyz v -> v (N xyz)")
        U = einops.rearrange(_U, "(J i j) (N xyz) -> J i j N xyz", i=3, j=3, xyz=3)

        _perFrameVerts = einops.rearrange(perFrameVerts, "b (N xyz) -> b N xyz", N = 5023)

        _jointRegressor = self.jointRegressor.unsqueeze(0).expand(self.nFrames, 5, 5023)
        joints = torch.einsum('b N c, b k N -> b k c', _perFrameVerts, _jointRegressor)

        rotatedJoints = (rotation @ joints.unsqueeze(-1)).squeeze(-1)

        # Modify the rotation to homogenous transformation
        transform = F.pad(rotation, [0,1,0,1,0,0,0,0]) # (B, 3N, 4, 4)
        transform[..., -1, -1] = 1
        transform[..., :3, -1] = joints - rotatedJoints # add translation
        weightedTransforms = torch.einsum('b K i j, N K -> b N i j', transform, self.skinningWt)
        vertsHomo = F.pad(_perFrameVerts, [0, 1], value = 1)
        vertsRotated = torch.einsum('b N i j, b N j -> b N i', weightedTransforms, vertsHomo)[..., :3] # (B, 3N, 3)
        return vertsRotated

    def LSB(self, rotation=True, identity=False):
        shapeOffset = self.blendShape()
        exprOffset = self.blendExpr()

        if identity:
            exprOffset *= 0.
            shapeOffset *= 0.

        perFrameVerts = self.verts.unsqueeze(0).expand(self.nFrames, 5023, 3).view(self.nFrames, -1) + shapeOffset.view(self.nFrames, -1) + exprOffset.view(self.nFrames, -1)
        rotatedPerFrameVerts = self.identityRotation(perFrameVerts)

        transform = torch.eye(4, device=self.device)[None]
        transform = transform.repeat(self.rot.shape[0], 1, 1)

        transform[:, :3, :3] = batch_rodrigues(self.rot)
        transform[:, :3, 3] = self.tra
        self.se3 = transform
        
        if not rotation:
            transform = torch.eye(4, device=self.device)[None]
            transform = transform.repeat(self.rot.shape[0], 1, 1)
            
        perFrameVerts = transform_vertices(transform, rotatedPerFrameVerts)
        perFrameVerts[..., 2] *= -1

        return perFrameVerts

    def sample_uv(self, map):
        # N,H,W,C to N,C,H,W
        map = map[...,:3].permute(0,3,1,2)
    
        grid = self.vertsUV.clone()
        grid[..., 0] = -2 * grid[..., 0] + 1  # u: [0,1] -> [1,-1]
        grid[..., 1] = 1 - grid[..., 1] * 2  # v: [0,1] -> [1,-1] (flip for image coords)
        
        # sample each vertex from uv space map through uv coordinates
        vert_values = F.grid_sample(
            map, 
            grid[None, :, None, :2].repeat(map.shape[0], 1, 1, 1), 
            align_corners=False,
            padding_mode='border',
        )
        vert_values = vert_values.permute(0, 2, 1, 3)[..., 0]
        return vert_values

    def renderVertsAnimation(
        self,
        savePath,
        vertsTexture = False,
        customVerts=None,
        resolution=256,
        dist=0.2
    ):
        
        if customVerts is not None:
            perFrameVerts = customVerts
        elif self.seq is not None:
            perFrameVerts = self.seq
        else:
            perFrameVerts = self.LSB()

        self.nFrames, _, _ = perFrameVerts.shape

        perFrameFaces = self.faces.unsqueeze(0).expand(self.nFrames, self.faces.shape[0], 3)
        if not vertsTexture:
            perVertsTexture = torch.ones((self.nFrames, 5023, 3), dtype=torch.float32, device=self.device)
        else: 
            perVertsTexture = torch.ones((self.nFrames, 5023, 3), dtype=torch.float32, device=self.device)

        headMesh = Meshes(verts=perFrameVerts, faces=perFrameFaces)
        headMesh.textures = TexturesVertex(verts_features=perVertsTexture)

        R, T = look_at_view_transform(dist=dist, elev=0, azim=0)
        cameras = PerspectiveCameras(
            device=self.device, 
            focal_length=self.focalLength[0, :].unsqueeze(0).expand(self.nFrames, 2),
            principal_point=self.principalPoint[0, :].unsqueeze(0).expand(self.nFrames, 2),
            R=R.expand(self.nFrames, 3, 3), T=T.expand(self.nFrames, 3),
            in_ndc=False, image_size=((1096, 997),))

        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=[[10.0, 10.0, 10.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        )

        images = renderer(headMesh)
        frames = images.cpu().numpy()  # shape: (B, H, W, 3)

        # Convert from [0, 1] floats to [0, 255] uint8 if needed:
        frames = (frames * 255).astype(np.uint8)

        # Save as a GIF with 30 frames per second:
        imageio.mimsave(savePath, frames, fps=30, quality=10) 
    
    def _get_mesh_ndc(
        self,
        meshes: Meshes,
        cameras: PerspectiveCameras,
    ) -> Meshes:
        eps = None
        verts_world = meshes.verts_padded()
        verts_view = cameras.get_world_to_view_transform().transform_points(
            verts_world, eps=eps
        )
        projection_trafo = cameras.get_projection_transform().compose(
            cameras.get_ndc_camera_transform()
        )
        verts_ndc = projection_trafo.transform_points(verts_view, eps=eps)
        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes.update_padded(new_verts_padded=verts_ndc)

        return meshes_ndc

    def renderAnimation(
        self,
        meshes,
        resolution=256,
        size=None,
        extri = None,
        intri = None,
        dist=0.2,
        elev=0.,
        azim=0.,
        unlit=False,
        image_size = 1080,
        light_elev=10.,
        light_azim=10,
        light_dist=3.0,
        cull_backfaces=True,
    ):
        self.nFrames, _, _ = meshes.verts_padded().shape
        centers = meshes.verts_padded().mean(dim=1)

        if is_invalid(getattr(self, "focalLength", None)):
            self.focalLength = torch.tensor([1700.0, 1700.0]).unsqueeze(0).to(self.device)

        if is_invalid(getattr(self, "principalPoint", None)):
            self.principalPoint = torch.tensor([image_size / 2, image_size / 2]).unsqueeze(0).to(self.device)

        if size is not None:
            resolution=size
        else:
            resolution = (resolution, resolution)

        if extri is not None:
            R, T = extri[:, :3, :3], extri[:, :3, 3]
        else:
            R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=centers, device=self.device)
            R = R.expand(self.nFrames, 3, 3)
            T = T.expand(self.nFrames, 3)

        if intri is not None:
            if len(intri.shape) == 3:
                fl = intri[..., [0,1],[0,1]].to(self.device)
                pp = intri[..., [0,1],[2,2]].to(self.device)
            else:
                fl = intri[[0,1],[0,1]][None].repeat(self.nFrames,1).to(self.device)
                pp = intri[[0,1],[2,2]][None].repeat(self.nFrames,1).to(self.device)
        else:
            fl = self.focalLength[0, :].unsqueeze(0).expand(self.nFrames, 2)
            pp = self.principalPoint[0, :].unsqueeze(0).expand(self.nFrames, 2)

        H, W = resolution
        img_size = torch.tensor([[W, H]] * len(R), dtype=torch.int, device=R.device)
        scale = img_size.to(R).min(dim=1, keepdim=True)[0] / 2.0
        scale = scale.expand(-1, 2)
        c0 = img_size / 2.0
        focal_pytorch3d = fl / scale
        p0_pytorch3d = -(pp - c0) / scale

        # R_pytorch3d = R.clone()
        # T_pytorch3d = T.clone()

        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = T.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1

        cameras = PerspectiveCameras(
            device=self.device, 
            focal_length=focal_pytorch3d,
            principal_point=p0_pytorch3d,  # Fixed: use principal_pytorch3d
            R=R_pytorch3d, T=T_pytorch3d,
            image_size=img_size,
        )

        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if znear is None else znear / 2

        raster_settings = RasterizationSettings(
            image_size=(H, W),  # Use a single tuple for all frames
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=cameras is not None,
            cull_backfaces=cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=False,
        )
        import math
        verts_p = meshes.verts_padded()
        mc = verts_p.mean(dim=1).mean(dim=0)
        avg_normal = meshes.verts_normals_padded().mean(dim=1).mean(dim=0)
        avg_normal = avg_normal / (avg_normal.norm() + 1e-8)

        up_candidate = torch.tensor([0.0, -1.0, 0.0], device=avg_normal.device)
        if torch.abs(torch.dot(avg_normal, up_candidate)) > 0.99:
            up_candidate = torch.tensor([1.0, 0.0, 0.0], device=avg_normal.device)
        right = torch.linalg.cross(up_candidate, avg_normal, dim=-1)
        right = right / (right.norm() + 1e-8)
        up = torch.linalg.cross(avg_normal, right, dim=-1)
        up = up / (up.norm() + 1e-8)

        el_rad = math.radians(light_elev)
        az_rad = math.radians(light_azim)
        offset = (
            light_dist * math.cos(el_rad) * math.cos(az_rad) * avg_normal
            + light_dist * math.cos(el_rad) * math.sin(az_rad) * right
            + light_dist * math.sin(el_rad) * up
        )
        light_pos = mc + offset

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        if unlit:
            shader = VertexColorShader(device=self.device, cameras=cameras)
        else:
            lights = PointLights(
                device=self.device,
                location=[light_pos.tolist()],
            )
            shader = SafeSoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
            )

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        images = renderer(meshes)
        return images

    def convertUV(self, customSeq=None, rotation=False, norm=False):
        #  Load and scale sequence
        seq = None
        if customSeq is not None: seq = customSeq
        elif self.seq is None: seq = self.LSB()
        else: seq = self.seq
        # seq = self.scaleIsotropic(seq)
        self.nFrames, _, _ = seq.shape

        if rotation:
            gMin = torch.tensor([-0.1776, -0.2074, -1.0983], dtype=torch.float32).to(self.device)
            gMax = torch.tensor([ 0.1772,  0.1642, -0.6894], dtype=torch.float32).to(self.device)
        else:
            gMin = torch.tensor([-0.1124, -0.1944, -0.1546], dtype=torch.float32).to(self.device)
            gMax = torch.tensor([0.1093, 0.1337, 0.0895], dtype=torch.float32).to(self.device)

        _seq = (seq - gMin) / (gMax - gMin) if norm else seq

        if _seq.shape[-1] == 3:
            r, g, b = _seq[:,:,0], _seq[:,:,1], _seq[:,:,2]
        else:
            r, g, b = _seq[:,:,0], _seq[:,:,1], torch.zeros(_seq[:,:,0].shape, device=self.device)

        colours = torch.stack([r,g,b], dim=2)

        nColours = colours[:, self.faces3d.view(-1), :].view(self.nFrames, self.nFaces, 3, 3)
        nUVFeatures = torch.zeros(self.nFrames, self.vertsUV.shape[0], 3).to(self.device)
        nUVFeatures[:, self.facesUV.view(-1), ...] = nColours.view(self.nFrames, self.nFaces * 3, 3)

        # Give each vertex a slightly unique z-depth to avoid z-fighting in overlapping UV regions
        # Small variation (0 to 0.01) based on vertex index, all vertices still essentially at z=1
        # vertsUV3d = torch.stack([self.vertsUV[:, 0], self.vertsUV[:, 1], torch.ones(self.vertsUV.shape[0]).to(self.device)], dim=1) 
        z_depth = 1.0 + torch.arange(self.vertsUV.shape[0], device=self.device, dtype=torch.float32) * 1e-5
        vertsUV3d = torch.stack([self.vertsUV[:, 0], self.vertsUV[:, 1], z_depth], dim=1) 
        self.vertsUV3d = vertsUV3d.unsqueeze(0).expand(self.nFrames, -1, -1)
        self.facesUV3d = self.facesUV.unsqueeze(0).expand(self.nFrames, -1, -1)
        uvTexture = TexturesVertex(verts_features=nUVFeatures)

        return Meshes(verts=self.vertsUV3d, faces=self.facesUV3d, textures=uvTexture)

    def fillBlackPixels(self, images, max_iter=50, threshold=1e-8, eps=1e-6):
        """
        images: (B, C, H, W) tensor. Black/missing pixels are all-zeros across channels.
        Fills black pixels by averaging non-black neighbors (3x3) iteratively.
        Works even if some channels are negative.
        """
        assert images.dim() == 4, "Expected (B, C, H, W)"
        B, C, H, W = images.shape
        device, dtype = images.device, images.dtype

        result = images.clone()

        # A pixel is "non-black" if ANY channel is nonzero (within threshold)
        non_black_mask = (result.abs() > threshold).any(dim=1, keepdim=True).float()  # (B,1,H,W)

        # 3x3 ones kernel per channel for grouped convs
        k = torch.ones((C, 1, 3, 3), device=device, dtype=dtype)

        it = 0
        while (non_black_mask == 0).any() and it < max_iter:
            # Sum of neighbor values, but only count non-black pixels
            # (Multiply by mask to zero-out black pixels before summing)
            sum_vals = F.conv2d(result * non_black_mask, k, padding=1, groups=C)  # (B,C,H,W)

            # Count of non-black neighbors per channel
            # Expand mask to channels for grouped conv
            ch_mask = non_black_mask.expand(-1, C, -1, -1)  # (B,C,H,W)
            count = F.conv2d(ch_mask, k, padding=1, groups=C)  # (B,C,H,W)

            # Average of *non-black* neighbors
            new_vals = sum_vals / (count + eps)

            # Fill where currently black but has at least one non-black neighbor
            fill_mask = (non_black_mask == 0) & (count > 0)  # (B,1,H,W)
            result = torch.where(fill_mask.expand_as(result), new_vals, result)

            # Update mask
            non_black_mask = (result.abs() > threshold).any(dim=1, keepdim=True).float()
            it += 1

        return result


    def renderUV(self, uvMesh, savePath=None, resolution=256, fill=True):
        uvBlendParams = BlendParams(background_color=(0.0, 0.0, 0.0))

        # UV mesh vertices are in [0, 1] range
        # T=[-0.5, -0.5, 0] centers the mesh, focal_length=2.0 scales to fill NDC [-1,1]
        uvCameras = OrthographicCameras(
            device=self.device,
            R=torch.eye(3).unsqueeze(0),               # No rotation
            T=torch.tensor([[-0.5, -0.5, 0.0]], dtype=torch.float32, device=self.device),
            focal_length=2.0,
        )

        uvRasterSettings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=4,  # Allow multiple faces per pixel to handle overlapping regions
            cull_backfaces=False,  # Don't cull - UV faces might be wound differently
        )
        
        # Use VertexColorShader to preserve exact vertex values without lighting
        # This simply interpolates vertex colors using barycentric coordinates
        uvRenderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=uvCameras,
                raster_settings=uvRasterSettings
            ),
            shader=VertexColorShader(
                device=self.device,
                cameras=uvCameras,
                blend_params=uvBlendParams
            )
        )
        
        perFrameUV = uvRenderer(uvMesh)
        
        if fill:
            smoothedPerFrameUV = self.fillBlackPixels(perFrameUV.permute(0,3,1,2)).permute(0,2,3,1)
        else:
            smoothedPerFrameUV = perFrameUV

        if savePath is not None:
            frames_rgb = (smoothedPerFrameUV[..., :3] * 255).byte().cpu().numpy()  
            imageio.mimsave(savePath, frames_rgb, fps=30, quality=10) 

        return smoothedPerFrameUV
    
    def get_uv_animation(self, uvMesh, savePath=None, resolution=256, sample_frames=50, fill=True, scale=False):
        verts_all  = uvMesh.verts_padded()           # [B, V, 3]
        faces_all  = uvMesh.faces_padded()           # [B, F, 3]
        feat_all   = uvMesh.textures.verts_features_padded()  # [B, V, C]

        if scale:
            feat_min = feat_all.amin(dim=1, keepdim=True)  # [B, 1, 3]
            feat_max = feat_all.amax(dim=1, keepdim=True)  # [B, 1, 3]
            feat_all = (feat_all - feat_min) / (feat_max - feat_min + 1e-8)

        B = verts_all.shape[0]   # total number of frames

        if B > sample_frames:
            verts_all = verts_all[:sample_frames, ...]
            faces_all = faces_all[:sample_frames, ...]
            feat_all = feat_all[:sample_frames, ...]
            B = sample_frames

        out = []
        chunk_size = 400
        
        def render_chunk(start, end):
            """Helper to build a sub‐Meshes and render it."""
            chunk = Meshes(
                verts=verts_all[start:end],
                faces=faces_all[start:end],
                textures=TexturesVertex(verts_features=feat_all[start:end])
            )
            return self.renderUV(chunk, savePath=None, fill=fill, resolution=resolution)

        try:
            # Render in chunks of 400
            for start_idx in range(0, B, chunk_size):
                end_idx = min(start_idx + chunk_size, B)
                out.append(render_chunk(start_idx, end_idx))
                
        except RuntimeError as e:
            # capture which chunk failed
            start_idx = len(out) * chunk_size
            end_idx = min(start_idx + chunk_size, B)
            print(f"→ [Flame.get_uv_animation] Illegal memory access during frames {start_idx}–{end_idx}")
            print(savePath)
            print(e)
            # return whatever we've got so far
            return torch.cat(out, dim=0) if out else None

        out = torch.cat(out, dim=0)

        if savePath is not None:
            frames_rgb = (out[..., :3] * 255).byte().cpu().numpy()  
            imageio.mimsave(savePath, frames_rgb, fps=73, quality=10) 
            return out
        else:
            return out[..., :3]

    def sampleFromUV(self, perFrameUV, savePath=None, resolution=256, returnMesh=False, frames=50, fill=True, scale=True):
        N = perFrameUV.shape[0]
        sampledUV = self.sample_uv(perFrameUV)

        sampledUVMesh = Meshes(verts=self.vertsUV3d[:N], faces=self.facesUV3d[:N], textures=TexturesVertex(verts_features=sampledUV))

        if savePath is not None: self.get_uv_animation(sampledUVMesh, savePath=savePath, resolution=resolution, sample_frames=frames, fill=fill, scale=True)

        if returnMesh:
            return sampledUV, sampledUVMesh
        
        return sampledUV
    

    def sampleTo3D(
            self, sampledUV, savePath=None, 
            dist=0.2, elev=0., azim=0., 
            extri=None, intri=None, scale=1, resolution=256, fps=73, size=None,
            norm=False, rotation=True, delta=None, denorm=None, returnVideo=False
        ):
        self.nFrames, nVerts, nChannel = sampledUV.shape 

        _sampledUV = sampledUV.clone()

        # index all faces in UV space from sampled UV image in 1 array  -->  (num frames, num faces, 3 vertices, position of each vertices) i.e. the faces  with verts position in 3d space instead of index
        faces3dFromUV = _sampledUV[:, self.facesUV.view(-1),:].reshape(self.nFrames,  -1, 3)

        # Get sampled sequence from sampled features 
        sampledSeq = torch.zeros(self.nFrames, 5023, 3).to(self.device)
        sampledSeq[:, self.faces3d.view(-1),:] = faces3dFromUV[:, :, :]

        if delta is None and norm:
            gMin = torch.tensor([-0.1776, -0.2074, -1.0983], dtype=torch.float32).to(self.device)
            gMax = torch.tensor([ 0.1772,  0.1642, -0.6894], dtype=torch.float32).to(self.device)

            perFrameVerts = (sampledSeq * (gMax - gMin) + gMin) * 3
        elif delta is not None: 
            perFrameVerts = delta[0:1,...] + sampledSeq
        else:
            perFrameVerts = sampledSeq  

        if extri is None:
            return perFrameVerts

        self.nFrames, _, _ = perFrameVerts.shape
        # perFrameVerts[..., 2] *= -1

        perFrameFaces = self.faces.unsqueeze(0).expand(self.nFrames, self.faces.shape[0], 3)
        perVertsTexture = torch.ones((self.nFrames, 5023, 3), dtype=torch.float32, device=self.device)

        headMesh = Meshes(verts=perFrameVerts, faces=perFrameFaces)
        headMesh.textures = TexturesVertex(verts_features=perVertsTexture)
                
        # if savePath is not None: self.renderAnimation(savePath, customVerts=sampledSeq * 2, dist=dist)
        verts_all  = headMesh.verts_padded() * scale            # [B, V, 3]
        faces_all  = headMesh.faces_padded()                    # [B, F, 3]
        feat_all   = headMesh.textures.verts_features_padded()  # [B, V, C]

        B = verts_all.shape[0]   # total number of frames
        out = []

        def render_chunk(start, end, dist=dist):
            """Helper to build a sub‐Meshes and render it."""
            chunk = Meshes(
                verts=verts_all[start:end],
                faces=faces_all[start:end],
                textures=TexturesVertex(verts_features=feat_all[start:end])
            )
            intri_chunk = intri[start:end] if intri is not None and intri.ndim == 3 else intri
            return self.renderAnimation(
                chunk, dist=dist, elev=elev, azim=azim, 
                resolution=resolution, size=size,
                extri=extri[start:end], intri=intri_chunk,
            )

        fail_start, fail_end = 0, B
        try:
            if B <= 491:
                # one‐shot render
                fail_start, fail_end = 0, B
                out.append(render_chunk(0, B, dist=dist))
            else:
                # first 491 frames
                fail_start, fail_end = 0, 491
                out.append(render_chunk(0, 491, dist=dist))
                # remaining frames
                fail_start, fail_end = 491, B
                out.append(render_chunk(491, B, dist=dist))
        except RuntimeError as e:
            # capture which chunk failed
            print(f"→ [Flame.sampleTo3D] Illegal memory access during frames {fail_start}–{fail_end}")
            print(savePath)
            print(e)
            # you could decide to continue on the next chunk or break outright
            # here we just stop and return whatever we've got so far
            return torch.cat(out, dim=0) if out else None

        out = torch.cat(out, dim=0)

        if savePath is not None:
            frames_rgb = (out[..., :3] * 255).byte().cpu().numpy()  
            imageio.mimsave(savePath, frames_rgb, fps=fps, quality=10)
            if returnVideo:
                return perFrameVerts, frames_rgb
            return perFrameVerts
        else:
            return perFrameVerts
        

    def render_chunk(
        self,
        verts,
        size,
        savePath,
        fps=73,
        extri=None,
        intri=None,
        resolution=256,
        scale=1.0,
        dist=0.6,
        azim=30,
        elev=30,
        light_elev=45.,
        light_azim=0.,
        light_dist=3.0,
        *,
        flip_normals: bool = False,
        flip_normals_z: bool = False,
    ):
        self.nFrames = verts.shape[0]

        perFrameFaces = self.faces.unsqueeze(0).expand(self.nFrames, self.faces.shape[0], 3)

        headMesh = Meshes(verts=verts, faces=perFrameFaces)
        vn = headMesh.verts_normals_padded()
        vn[..., 2] = -vn[..., 2]  # flip Z so camera-facing normals → blue
        if flip_normals:
            # Full sign flip on top of the Z-flip above. Net effect on the
            # rendered (vn * 0.5 + 0.5) texture is the colour complement
            # (e.g. blue front-face -> orange front-face), which makes A/B
            # comparison across pred-mesh columns visually obvious.
            vn = -vn
        if flip_normals_z:
            # Z-only flip (orthogonal to `flip_normals`, which is the full
            # sign flip).  Toggles the default Z-flip applied above, so
            # camera-facing normals come out coloured by the *unflipped* Z
            # sign instead of the default convention.  Useful when an
            # external pipeline expects vert normals in OpenCV cam-space
            # (+Z into scene, away from camera) rather than the default
            # "blue-on-front" pyt3d-style mapping.
            vn = vn.clone()
            vn[..., 2] = -vn[..., 2]
        perVertsTexture = (vn * 0.5 + 0.5).clamp(0, 1)
        headMesh.textures = TexturesVertex(verts_features=perVertsTexture)

        if extri is None:
            extri = torch.eye(4, device=self.device)[None].repeat(self.nFrames,1,1)

        verts_all  = headMesh.verts_padded() * scale            # [B, V, 3]
        faces_all  = headMesh.faces_padded()                    # [B, F, 3]
        feat_all   = headMesh.textures.verts_features_padded()  # [B, V, C]

        B = verts_all.shape[0]   # total number of frames
        out = []

        def render_chunk(start, end, dist=dist):
            """Helper to build a sub‐Meshes and render it."""
            chunk = Meshes(
                verts=verts_all[start:end],
                faces=faces_all[start:end],
                textures=TexturesVertex(verts_features=feat_all[start:end])
            )
            intri_chunk = intri[start:end] if intri is not None and intri.ndim == 3 else intri
            return self.renderAnimation(
                chunk, dist=dist, elev=elev, azim=azim, 
                resolution=resolution, size=size,
                extri=extri[start:end], intri=intri_chunk,
                light_elev=light_elev, light_azim=light_azim,
                light_dist=light_dist,
                unlit=True,
            )

        fail_start, fail_end = 0, B
        try:
            if B <= 491:
                # one‐shot render
                fail_start, fail_end = 0, B
                out.append(render_chunk(0, B, dist=dist))
            else:
                # first 491 frames
                fail_start, fail_end = 0, 491
                out.append(render_chunk(0, 491, dist=dist))
                # remaining frames
                fail_start, fail_end = 491, B
                out.append(render_chunk(491, B, dist=dist))
        except RuntimeError as e:
            # capture which chunk failed
            print(f"→ [Flame.render_chunk] Illegal memory access during frames {fail_start}–{fail_end}")
            print(savePath)
            print(e)
            # you could decide to continue on the next chunk or break outright
            # here we just stop and return whatever we've got so far
            return torch.cat(out, dim=0) if out else None

        out = torch.cat(out, dim=0)

        if savePath is not None:
            frames_rgb = (out[..., :3] * 255).byte().cpu().numpy()  
            imageio.mimsave(savePath, frames_rgb, fps=fps, quality=10)
        
        # Return rendered images (B, H, W, 4) with values in [0, 1]
        return out
            
    def get_3d_animation(self, perFrameVerts, savePath, perVertsTexture=None, resolution=256, dist=0.1, elev=0.0, azim=0.0, scale=1, fps=72):

        self.nFrames, _, _ = perFrameVerts.shape

        perFrameFaces = self.faces.unsqueeze(0).expand(self.nFrames, self.faces.shape[0], 3)

        if perVertsTexture is None:
            perVertsTexture = torch.ones((self.nFrames, 5023, 3), dtype=torch.float32, device=self.device)
        else: 
            perVertsTexture.to(device=self.device, dtype=torch.float32)
            # v_min = perVertsTexture.min()
            # v_max = perVertsTexture.max()
            # perVertsTexture = (perVertsTexture - v_min) / (v_max - v_min)
            # print(perVertsTexture)

        headMesh = Meshes(verts=perFrameVerts, faces=perFrameFaces)
        headMesh.textures = TexturesVertex(verts_features=perVertsTexture)
                
        # if savePath is not None: self.renderAnimation(savePath, customVerts=sampledSeq * 2, dist=dist)
        verts_all  = headMesh.verts_padded() * scale            # [B, V, 3]
        faces_all  = headMesh.faces_padded()                    # [B, F, 3]
        feat_all   = headMesh.textures.verts_features_padded()  # [B, V, C]

        B = verts_all.shape[0]   # total number of frames
        out = []

        def render_chunk(start, end, dist=dist):
            """Helper to build a sub‐Meshes and render it."""
            chunk = Meshes(
                verts=verts_all[start:end],
                faces=faces_all[start:end],
                textures=TexturesVertex(verts_features=feat_all[start:end])
            )
            return self.renderAnimation(chunk, dist=dist, elev=elev, azim=azim, resolution=resolution)

        fail_start, fail_end = 0, B
        try:
            if B <= 491:
                # one‐shot render
                fail_start, fail_end = 0, B
                out.append(render_chunk(0, B, dist=dist))
            else:
                # first 491 frames
                fail_start, fail_end = 0, 491
                out.append(render_chunk(0, 491, dist=dist))
                # remaining frames
                fail_start, fail_end = 491, B
                out.append(render_chunk(491, B, dist=dist))
        except RuntimeError as e:
            # capture which chunk failed
            print(f"→ [Flame.get_3d_animation] Illegal memory access during frames {fail_start}–{fail_end}")
            print(savePath)
            print(e)
            # you could decide to continue on the next chunk or break outright
            # here we just stop and return whatever we've got so far
            return torch.cat(out, dim=0) if out else None

        out = torch.cat(out, dim=0)

        if savePath is not None:
            frames_rgb = (out[..., :3] * 255).byte().cpu().numpy()  
            imageio.mimsave(savePath, frames_rgb, fps=fps, quality=10) 
            return perFrameVerts
        else:
            return perFrameVerts
        
        
    def batch_uv(self, paths, resolution=256, sample_frames=50, rotation=True):
        out = []
        min_frame = 10000000
        for path in paths:
            self.loadSequence(path)
            x = self.LSB(rotation=rotation) 
            frame, _, _ = x.shape    
            if frame < min_frame: min_frame = frame                                       
            uvMesh = self.convertUV(rotation=rotation)                     
            uv = self.get_uv_animation(uvMesh, resolution=resolution, sample_frames=sample_frames) 
            out.append(uv[..., :3].unsqueeze(0))
        out = [item[:, :min_frame, ...] for item in out]
        return torch.concat(out, dim=0)
