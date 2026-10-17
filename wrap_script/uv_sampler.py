import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import warnings
from pathlib import Path
from collections import namedtuple
from typing import Optional, List, Tuple, Dict

import numpy as np

# Named tuples matching PyTorch3D's structure
_Faces = namedtuple("_Faces", ["verts_idx", "normals_idx", "textures_idx", "materials_idx"])
_Aux = namedtuple("_Aux", ["normals", "verts_uvs", "material_colors", "texture_images", "texture_atlas"])


def load_obj(
    f,
    load_textures: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, _Faces, _Aux]:
    """
    Load a mesh from a .obj file.
    
    Args:
        f: Path to .obj file (str or Path)
        load_textures: Whether to load material/texture info (not implemented here)
        device: Device to load tensors on ('cpu' or 'cuda')
    
    Returns:
        verts: FloatTensor of shape [V, 3]
        faces: NamedTuple with fields:
            - verts_idx: LongTensor [F, 3] - vertex indices
            - normals_idx: LongTensor [F, 3] or None - normal indices
            - textures_idx: LongTensor [F, 3] or None - UV indices
            - materials_idx: LongTensor [F] or None - material indices per face
        aux: NamedTuple with fields:
            - normals: FloatTensor [N, 3] or None
            - verts_uvs: FloatTensor [T, 2] or None - UV coordinates
            - material_colors: None (not implemented)
            - texture_images: None (not implemented)
            - texture_atlas: None (not implemented)
    """
    
    # Get data directory for materials
    data_dir = "./"
    if isinstance(f, (str, Path)):
        data_dir = os.path.dirname(f)
    
    # Parse the OBJ file
    (
        verts,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    ) = _parse_obj(f, data_dir)
    
    # Convert to tensors
    verts = _make_tensor(verts, cols=3, dtype=torch.float32, device=device)
    normals = _make_tensor(normals, cols=3, dtype=torch.float32, device=device)
    verts_uvs = _make_tensor(verts_uvs, cols=2, dtype=torch.float32, device=device)
    
    # Format face indices (convert from 1-indexed to 0-indexed)
    faces_verts_idx = _format_faces_indices(
        faces_verts_idx, verts.shape[0], device=device
    )
    
    # Handle normals and textures if present
    if len(faces_normals_idx):
        faces_normals_idx = _format_faces_indices(
            faces_normals_idx, normals.shape[0], device=device, pad_value=-1
        )
    else:
        faces_normals_idx = None
    
    if len(faces_textures_idx):
        faces_textures_idx = _format_faces_indices(
            faces_textures_idx, verts_uvs.shape[0], device=device, pad_value=-1
        )
    else:
        faces_textures_idx = None
    
    if len(faces_materials_idx):
        faces_materials_idx = torch.tensor(
            faces_materials_idx, dtype=torch.int64, device=device
        )
    else:
        faces_materials_idx = None
    
    # Create named tuples
    faces = _Faces(
        verts_idx=faces_verts_idx,
        normals_idx=faces_normals_idx,
        textures_idx=faces_textures_idx,
        materials_idx=faces_materials_idx,
    )
    
    aux = _Aux(
        normals=normals if len(normals) else None,
        verts_uvs=verts_uvs if len(verts_uvs) else None,
        material_colors=None,  # Not implemented
        texture_images=None,   # Not implemented
        texture_atlas=None,    # Not implemented
    )
    
    return verts, faces, aux


def _parse_obj(f, data_dir: str):
    """
    Parse OBJ file and extract all data.
    
    Returns:
        Tuple of lists containing vertices, normals, UVs, face indices, etc.
    """
    verts, normals, verts_uvs = [], [], []
    faces_verts_idx, faces_normals_idx, faces_textures_idx = [], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = None
    materials_idx = -1
    
    # Read file
    if isinstance(f, (str, Path)):
        with open(f, 'r') as file:
            lines = file.readlines()
    else:
        lines = f.readlines()
    
    # Strip whitespace
    lines = [line.strip() for line in lines]
    
    # Parse each line
    for line in lines:
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        tokens = line.split()
        if not tokens:
            continue
        
        # Material library
        if line.startswith("mtllib"):
            if len(tokens) < 2:
                raise ValueError("material file name is not specified")
            mtl_path = line[len(tokens[0]):].strip()
            mtl_path = os.path.join(data_dir, mtl_path)
            if not os.path.exists(mtl_path):
                warnings.warn(f"Mtl file does not exist: {mtl_path}")
        
        # Use material
        elif tokens[0] == "usemtl" and len(tokens) > 1:
            material_name = tokens[1]
            if material_name not in material_names:
                material_names.append(material_name)
                materials_idx = len(material_names) - 1
            else:
                materials_idx = material_names.index(material_name)
        
        # Vertex
        elif line.startswith("v "):
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                raise ValueError(f"Vertex {vert} does not have 3 values. Line: {line}")
            verts.append(vert)
        
        # Texture coordinate
        elif line.startswith("vt "):
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(f"Texture {tx} does not have 2 values. Line: {line}")
            verts_uvs.append(tx)
        
        # Normal
        elif line.startswith("vn "):
            norm = [float(x) for x in tokens[1:4]]
            if len(norm) != 3:
                raise ValueError(f"Normal {norm} does not have 3 values. Line: {line}")
            normals.append(norm)
        
        # Face
        elif line.startswith("f "):
            _parse_face(
                line,
                tokens,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )
    
    return (
        verts,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    )


def _parse_face(
    line: str,
    tokens: List[str],
    material_idx: int,
    faces_verts_idx: List,
    faces_normals_idx: List,
    faces_textures_idx: List,
    faces_materials_idx: List,
) -> None:
    """
    Parse a face line from OBJ file.
    
    Face format: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
    or: f v1//vn1 v2//vn2 v3//vn3 (no texture)
    or: f v1/vt1 v2/vt2 v3/vt3 (no normal)
    or: f v1 v2 v3 (only vertices)
    """
    face = tokens[1:]
    face_list = [f.split("/") for f in face]
    face_verts = []
    face_normals = []
    face_textures = []
    
    for vert_props in face_list:
        # Vertex index (always present)
        face_verts.append(int(vert_props[0]))
        
        if len(vert_props) > 1:
            # Texture index (may be empty: v//vn)
            if vert_props[1] != "":
                face_textures.append(int(vert_props[1]))
            
            if len(vert_props) > 2:
                # Normal index
                face_normals.append(int(vert_props[2]))
            
            if len(vert_props) > 3:
                raise ValueError(
                    f"Face vertices can only have 3 properties. "
                    f"Face vert {vert_props}, Line: {line}"
                )
    
    # Fill missing indices with -1 (padding)
    if len(face_normals) > 0:
        if len(face_verts) != len(face_normals):
            raise ValueError(
                f"Face {face} is inconsistent. Line: {line}"
            )
    else:
        face_normals = [-1] * len(face_verts)
    
    if len(face_textures) > 0:
        if len(face_verts) != len(face_textures):
            raise ValueError(
                f"Face {face} is inconsistent. Line: {line}"
            )
    else:
        face_textures = [-1] * len(face_verts)
    
    # Triangulate faces with more than 3 vertices
    # Split into triangles using fan triangulation
    for i in range(len(face_verts) - 2):
        faces_verts_idx.append((face_verts[0], face_verts[i + 1], face_verts[i + 2]))
        faces_normals_idx.append((face_normals[0], face_normals[i + 1], face_normals[i + 2]))
        faces_textures_idx.append((face_textures[0], face_textures[i + 1], face_textures[i + 2]))
        faces_materials_idx.append(material_idx)


def _make_tensor(
    data: List,
    cols: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """
    Convert list of lists to tensor.
    """
    if not data:
        return torch.tensor([], dtype=dtype, device=device).reshape(0, cols)
    
    return torch.tensor(data, dtype=dtype, device=device)


def _format_faces_indices(
    faces_indices: List[Tuple],
    max_index: int,
    device: str,
    pad_value: Optional[int] = None,
) -> torch.Tensor:
    """
    Format face indices: convert to 0-based indexing and handle negative indices.
    
    OBJ files use 1-based indexing. Negative indices count from the end.
    """
    if not faces_indices:
        return torch.tensor([], dtype=torch.int64, device=device).reshape(0, 3)
    
    faces_indices = torch.tensor(faces_indices, dtype=torch.int64, device=device)
    
    if pad_value is not None:
        mask = faces_indices.eq(pad_value).all(dim=-1)
    
    # Convert to 0-based indexing (OBJ is 1-based)
    faces_indices[faces_indices > 0] -= 1
    
    # Handle negative indices (count from end)
    faces_indices[faces_indices < 0] += max_index
    
    if pad_value is not None:
        faces_indices[mask] = pad_value
    
    return _check_faces_indices(faces_indices, max_index, pad_value)


def _check_faces_indices(
    faces_indices: torch.Tensor,
    max_index: int,
    pad_value: Optional[int] = None,
) -> torch.Tensor:
    """
    Check that face indices are valid.
    """
    if pad_value is not None:
        mask = faces_indices.eq(pad_value).all(dim=-1)
        # Check all faces that are not padded
        if (~mask).any():
            non_padded = faces_indices[~mask]
            if (non_padded < 0).any() or (non_padded >= max_index).any():
                warnings.warn("Faces have invalid indices")
    else:
        if (faces_indices < 0).any() or (faces_indices >= max_index).any():
            warnings.warn("Faces have invalid indices")
    
    return faces_indices

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
    return x_norm
    
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

class UVSampler(nn.Module):
    def __init__(
            self,
            obj_dir : str = "data/flame/", 
            device : str = "cuda"
        ):
        super(UVSampler, self).__init__() # Initialize the nn.Module

        self.device = device

        verts, faces, aux = load_obj(obj_dir + "flame_uv_flowface.obj", device=self.device)
        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)

        self.facesUV, self.vertsUV = faces.textures_idx.to(self.device), aux.verts_uvs.to(self.device)
        self.faces3d, self.verts3d = faces_idx, verts
        self.nFaces, _ = self.faces3d.shape

        self.identity_pose = torch.load(obj_dir + "identity_pose.pt").to(self.device)
        self.vertex_weights = torch.tensor(np.load("data/flame/vertex_weights.npy")).to(self.device)
        self.vertex_weights = self.vertex_weights[None, :, None]
        _vertex_weights = self.vertex_weights[:, self.faces3d.view(-1), :].view(1, self.nFaces, 3,  1)
        self.uv_weights = torch.zeros(1, self.vertsUV.shape[0], 1).to(self.device)
        self.uv_weights[:, self.facesUV.view(-1), ...] = _vertex_weights.view(1, self.nFaces * 3, 1)

        self.to(device)

    def sample_uv(self, map):
        # N,H,W,C to N,C,H,W
        N, H, W, C = map.shape
        map = map[...,:C].permute(0,3,1,2)
    
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
    
    def sampleFromUV(self, perFrameUV):
        sampledUV = self.sample_uv(perFrameUV)
        
        return sampledUV
    
    def sampleTo3D(self, sampledUV, dim=3, offset=False):
        self.nFrames, nVerts, nChannel = sampledUV.shape 

        _sampledUV = sampledUV.clone()

        # index all faces in UV space from sampled UV image in 1 array  -->  (num frames, num faces, 3 vertices, position of each vertices) i.e. the faces  with verts position in 3d space instead of index
        faces3dFromUV = _sampledUV[:, self.facesUV.view(-1),:].reshape(self.nFrames,  -1, dim)

        # Get sampled sequence from sampled features 
        sampledSeq = torch.zeros(self.nFrames, 5023, dim).to(self.device) 
        sampledSeq[:, self.faces3d.view(-1),:] = faces3dFromUV[:, :, :]

        if offset:
            perFrameVerts = self.identity_pose + sampledSeq
        else:
            perFrameVerts = sampledSeq

        return perFrameVerts