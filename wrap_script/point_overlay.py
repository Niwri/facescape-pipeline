"""Shared helpers for the "point overlay" renders: annotated (propagated)
landmarks vs. the same landmarks read off the wrapped mesh, drawn over one
camera image.

- Annotated points: propagated_landmarks.load_camera_standard_points()
  (sapiens2 frame_ merged with the propagated ear/nosebridge/undernose/
  eyebrow/chin group files -- what the wrap now triangulates from, and what
  the FaceView Cameras tab shows) plus skin_frame_ points, restricted to
  FaceView/whitelist.json's indices (+ any extra visible ones) and skin.
- Mesh points: index < 74 from neutral_correspondence_without_eyes_POT.json's
  point-on-triangle row for that index; index >= 74 (skin) from the exact
  vertex skin_landmarks.assign_global_indices() assigns it.

Style: annotated = filled dot in the Cameras tab's region color; wrapped mesh
= hollow ring in the same hue, blended toward white; a thin gray line joins
each pair so the error is visible at a glance. The top-left corner shows a
legend and the frame's mean pixel error.

Used by wrap_script/render_point_overlay.py (Nersemble) and
wrap_script_facescape/render_point_overlay_frame.py (FaceScape). Kept free of
backend/torch imports; render_landmarks.py (colors) and skin_landmarks.py are
loaded by file path so this works from either directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Also loaded by file path from wrap_script_facescape/ (render_point_overlay_frame.py,
# render_landmarks_facescape.py), where this folder isn't on sys.path: make nersemble_paths,
# which sits next to this file, importable either way.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import nersemble_paths  # noqa: E402
import contextlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

WRAP_SCRIPT_DIR = nersemble_paths.WRAP_SCRIPT_ROOT
# Nersemble's point-on-triangle file (what wrap_script's wrap uses).
POT_PATH = WRAP_SCRIPT_DIR / "neutral_correspondence_without_eyes_POT.json"
WHITELIST_PATH = nersemble_paths.FACEVIEW_ROOT / "whitelist.json"
NUM_STANDARD_LANDMARKS = 74


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_render_landmarks_colors() -> tuple[dict[int, tuple[int, int, int]], tuple[int, int, int]]:
    """INDEX_TO_COLOR / SKIN_COLOR (RGB) from render_landmarks.py, which mirrors
    FaceView's LandmarkOverlay.tsx. render_landmarks.py imports
    backend.app.core.config at module level only for its own CLI; a stub
    module stands in for it here if no backend is importable."""
    stubbed = False
    if "backend.app.core.config" not in sys.modules:
        try:
            importlib.import_module("backend.app.core.config")
        except ImportError:
            stub = types.ModuleType("backend.app.core.config")
            stub.get_settings = lambda: None
            for name in ("backend", "backend.app", "backend.app.core"):
                sys.modules.setdefault(name, types.ModuleType(name))
            sys.modules["backend.app.core.config"] = stub
            stubbed = True
    module = _load_module("wrap_script_render_landmarks", WRAP_SCRIPT_DIR / "render_landmarks.py")
    if stubbed:
        for name in ("backend.app.core.config", "backend.app.core", "backend.app", "backend"):
            sys.modules.pop(name, None)
    return dict(module.INDEX_TO_COLOR), tuple(module.SKIN_COLOR)


INDEX_TO_COLOR, SKIN_COLOR = _load_render_landmarks_colors()
_propagated = _load_module("wrap_script_propagated_landmarks", WRAP_SCRIPT_DIR / "propagated_landmarks.py")
load_camera_standard_points = _propagated.load_camera_standard_points
load_camera_skin_points = _propagated.load_camera_skin_points
_skin_landmarks = _load_module("wrap_script_skin_landmarks", WRAP_SCRIPT_DIR / "skin_landmarks.py")

# BGR, drawing constants.
LINE_COLOR = (170, 170, 170)
MESH_TINT = 0.5  # how far the wrapped-mesh ring color is blended toward white


def load_whitelist() -> set[int]:
    with open(WHITELIST_PATH) as f:
        return set(json.load(f))


def load_mesh_landmark_spec(pot_path: Path = POT_PATH) -> tuple[list[list[float]], dict[int, int]]:
    """(POT rows for standard indices 0-73, {skin global index: vertex index}).
    pot_path must be the POT file the dataset's own wrap used -- the datasets
    differ (e.g. FaceScape/Ava-256 moved ear indices 1/4/6/14 to other FLAME
    vertices); POT_PATH is Nersemble's."""
    with open(pot_path) as f:
        pot_rows = json.load(f)[:NUM_STANDARD_LANDMARKS]
    # assign_global_indices() prints a NOTE per duplicate vertex on every call.
    with contextlib.redirect_stdout(io.StringIO()):
        assigned = _skin_landmarks.assign_global_indices()
    skin_vertex_by_index = {global_idx: vertex_idx for global_idx, _region, vertex_idx in assigned}
    return pot_rows, skin_vertex_by_index


def mesh_landmarks_world(
    verts: np.ndarray, faces: np.ndarray, pot_rows: list[list[float]], skin_vertex_by_index: dict[int, int],
) -> dict[int, np.ndarray]:
    """{landmark index: world xyz} on the wrapped mesh. POT rows are
    [face, u, v] with corner weights (u, v, 1-u-v) for the face's vertices
    (0, 1, 2) -- (1,0) pins corner 0, (0,1) corner 1, (0,0) corner 2."""
    out: dict[int, np.ndarray] = {}
    for idx, (face_idx, u, v) in enumerate(pot_rows):
        a, b, c = faces[int(face_idx)]
        out[idx] = u * verts[a] + v * verts[b] + (1.0 - u - v) * verts[c]
    for idx, vertex_idx in skin_vertex_by_index.items():
        if vertex_idx < len(verts):
            out[idx] = verts[vertex_idx]
    return out


def project_points(xyz: np.ndarray, K: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """(N,3) world -> (N,2) pixels with a plain pinhole model (Rt world->cam)."""
    cam = xyz @ Rt[:3, :3].T + Rt[:3, 3]
    x = cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]
    y = cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]
    return np.stack([x, y], axis=-1)


def project_mesh_landmarks(mesh_world: dict[int, np.ndarray], K: np.ndarray, Rt: np.ndarray) -> dict[int, np.ndarray]:
    if not mesh_world:
        return {}
    indices = list(mesh_world)
    px = project_points(np.stack([mesh_world[i] for i in indices]), K, Rt)
    return dict(zip(indices, px))


def annotated_points_px(
    cam_dir: Path, frame_tag: str, width: int, height: int, visible_indices: set[int],
) -> dict[int, np.ndarray]:
    """{index: pixel xy} of the annotated points for one camera/frame, scaled
    to (width, height). Standard points outside visible_indices, and any
    point outside the image ([0,1] normalized -- includes sapiens2's -1
    "not detected" placeholders), are dropped."""
    out: dict[int, np.ndarray] = {}
    for point in load_camera_standard_points(cam_dir, frame_tag) or []:
        if point["index"] in visible_indices:
            out[point["index"]] = point
    for point in load_camera_skin_points(cam_dir, frame_tag) or []:
        out[point["index"]] = point
    return {
        idx: np.array([p["x"] * width, p["y"] * height])
        for idx, p in out.items()
        if 0.0 < p["x"] < 1.0 and 0.0 < p["y"] < 1.0
    }


def _color_bgr(index: int) -> tuple[int, int, int]:
    r, g, b = SKIN_COLOR if index >= NUM_STANDARD_LANDMARKS else INDEX_TO_COLOR.get(index, SKIN_COLOR)
    return (b, g, r)


def _tint(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(c + (255 - c) * MESH_TINT)) for c in bgr)


def point_radius(image: np.ndarray) -> int:
    return max(2, int(round(min(image.shape[:2]) / 200)))


def draw_point_overlay(
    image: np.ndarray, annotated: dict[int, np.ndarray], mesh_px: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, float]]:
    """Draws every index present in both dicts onto a copy of `image` (BGR).
    Returns (image, {index: pixel error} for every pair drawn) -- see
    split_errors() for standard/skin means."""
    out = image.copy()
    h, w = out.shape[:2]
    r = point_radius(out)
    indices = sorted(i for i in annotated if i in mesh_px and np.all(np.isfinite(mesh_px[i])))
    shift = 4  # sub-pixel precision for cv2 drawing
    scale = 1 << shift

    def pt(p: np.ndarray) -> tuple[int, int]:
        return int(round(p[0] * scale)), int(round(p[1] * scale))

    for i in indices:
        cv2.line(out, pt(annotated[i]), pt(mesh_px[i]), LINE_COLOR, 1, cv2.LINE_AA, shift)
    ring_thickness = 1 if r < 5 else 2
    for i in indices:
        cv2.circle(out, pt(mesh_px[i]), (r + 2) * scale, _tint(_color_bgr(i)), ring_thickness, cv2.LINE_AA, shift)
    for i in indices:
        color = _color_bgr(i)
        cv2.circle(out, pt(annotated[i]), (r + 1) * scale, (20, 20, 20), -1, cv2.LINE_AA, shift)
        cv2.circle(out, pt(annotated[i]), r * scale, color, -1, cv2.LINE_AA, shift)

    errors = {i: float(np.linalg.norm(annotated[i] - mesh_px[i])) for i in indices}
    _draw_legend(out, errors, r)
    return out, errors


def split_errors(errors: dict[int, float]) -> tuple[float | None, float | None]:
    """(mean error over standard indices, mean error over skin indices)."""
    std = [e for i, e in errors.items() if i < NUM_STANDARD_LANDMARKS]
    skin = [e for i, e in errors.items() if i >= NUM_STANDARD_LANDMARKS]
    return (float(np.mean(std)) if std else None), (float(np.mean(skin)) if skin else None)


def _draw_legend(image: np.ndarray, errors: dict[int, float], r: int) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, image.shape[1] / 1400), 1
    std_err, skin_err = split_errors(errors)
    fmt = lambda e: "-" if e is None else f"{e:.1f}px"  # noqa: E731
    lines = ["annotated", "wrapped mesh", f"mean err  std {fmt(std_err)}  skin {fmt(skin_err)}"]
    line_h = int(22 * scale / 0.4)
    box_w = int(250 * scale / 0.4)
    overlay = image.copy()
    cv2.rectangle(overlay, (4, 4), (4 + box_w, 8 + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)
    sample = (255, 255, 255)
    for row, text in enumerate(lines):
        y = 4 + line_h * (row + 1) - line_h // 4
        cx, cy = 14 + r, y - line_h // 4
        if row == 0:
            cv2.circle(image, (cx, cy), r, sample, -1, cv2.LINE_AA)
        elif row == 1:
            cv2.circle(image, (cx, cy), r + 2, _tint(sample), 1 if r < 5 else 2, cv2.LINE_AA)
        x_text = 14 + 2 * r + 8 if row < 2 else 10
        cv2.putText(image, text, (x_text, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
