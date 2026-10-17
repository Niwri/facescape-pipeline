"""Compact per-sequence export of wrapped_mesh.obj frames to a single .pt file.

Triggered once a pair is confirmed (see api/routes/labels.py's /labels/confirm
handler), so downstream consumers can load one small tensor bundle per
actor/sequence instead of re-parsing every frame's wrapped_mesh.obj. Every
wrapped_mesh.obj shares FLAME's fixed template topology -- same face count and
vertex count across every actor, sequence, and frame, since wrapping only moves
vertex positions -- so this saves faces once rather than repeating the same
(7800, 3) face array in every frame's .obj.

delete_exported_sequence() is the inverse, run when a "confirmed" sign-off is
reverted (see /labels/unconfirm-confirmed) so a stale export doesn't keep
being served for a pair that's no longer confirmed.

rename_wrapped_to_old_sequence()/rename_wrapped_to_old_facescape_pair() run
when an "unconfirmed" pair is sent back to "unreviewed" instead (see
/labels/unconfirm): renaming wrapped_mesh.obj -> old_wrapped_mesh.obj so
run_pipeline_batch_all.py's/run_pipeline_batch.py's own wrapped_mesh_exists()
skip-check doesn't mistake the old, presumably-rejected wrap for already-done
work the next time the batch wrap script runs this pair.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..core.config import get_settings
from . import facescape_invalid
from .facescape_coordinate_transform import (
    apply_similarity_to_verts,
    solve_facescape_inverse_transform,
)

log = logging.getLogger(__name__)


def _load_obj_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal parser for wrapped_mesh.obj's own simple format (only 'v x y z'
    and 'f i j k' lines, 1-based indices) -- avoids adding trimesh as a new
    backend dependency just for this one well-known, simple case."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                # Only the vertex-index component of each 'i[/vt[/vn]]' triplet.
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def export_wrapped_sequence(actor_id: str, sequence_id: str) -> Path | None:
    """Loads every frame's wrapped_mesh.obj for this actor/sequence pair and
    writes {faces: (F, 3) int64, verts: (N, V, 3) float32, frame_ids: (N,)
    int64} to final_wrapped_data_root/<actor>/<sequence>.pt.

    Never raises -- this runs as a side effect of a successful /labels/confirm,
    and a missing/partial export shouldn't be treated as the confirm action
    itself having failed. Returns the written path, or None if nothing was
    exported (logged either way)."""
    settings = get_settings()
    sequence_output_dir = settings.output_root / actor_id / sequence_id

    try:
        frame_dirs = sorted(
            (
                (int(p.name), p / "wrapped_mesh.obj")
                for p in sequence_output_dir.iterdir()
                if p.is_dir() and p.name.isdigit() and (p / "wrapped_mesh.obj").exists()
            ),
            key=lambda pair: pair[0],
        )
    except OSError as exc:
        log.warning("export_wrapped_sequence: cannot list %s: %s", sequence_output_dir, exc)
        return None

    if not frame_dirs:
        log.warning(
            "export_wrapped_sequence: no wrapped_mesh.obj found under %s for %s/%s",
            sequence_output_dir, actor_id, sequence_id,
        )
        return None

    faces: np.ndarray | None = None
    verts_by_frame: list[np.ndarray] = []
    frame_ids: list[int] = []
    for frame_id, obj_path in frame_dirs:
        try:
            frame_verts, frame_faces = _load_obj_vertices_faces(obj_path)
        except (OSError, ValueError) as exc:
            log.warning("export_wrapped_sequence: skipping unreadable %s: %s", obj_path, exc)
            continue

        if faces is None:
            faces = frame_faces
        elif faces.shape != frame_faces.shape or not np.array_equal(faces, frame_faces):
            log.warning(
                "export_wrapped_sequence: %s has a different face topology than "
                "frame %d -- skipping this frame (kept the first frame's faces)",
                obj_path, frame_dirs[0][0],
            )
            continue

        verts_by_frame.append(frame_verts)
        frame_ids.append(frame_id)

    if faces is None or not verts_by_frame:
        log.warning("export_wrapped_sequence: nothing usable under %s for %s/%s",
                     sequence_output_dir, actor_id, sequence_id)
        return None

    verts_stack = torch.from_numpy(np.stack(verts_by_frame, axis=0))  # (N, V, 3)
    faces_tensor = torch.from_numpy(faces)  # (F, 3)
    frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.int64)

    output_path = settings.final_wrapped_data_root / actor_id / f"{sequence_id}.pt"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"faces": faces_tensor, "verts": verts_stack, "frame_ids": frame_ids_tensor},
            output_path,
        )
    except OSError as exc:
        log.warning("export_wrapped_sequence: failed to write %s: %s", output_path, exc)
        return None

    log.info(
        "export_wrapped_sequence: wrote %s (%d frames, %d verts, %d faces)",
        output_path, verts_stack.shape[0], verts_stack.shape[1], faces_tensor.shape[0],
    )
    return output_path


def export_wrapped_facescape_pair(actor_id: str, expression_id: str) -> Path | None:
    """FaceScape counterpart to export_wrapped_sequence() above -- same {faces,
    verts, frame_ids}.pt format, but FaceScape has exactly one static
    wrapped_mesh.obj per actor/expression pair (no per-frame subdirectories,
    see wrap_script_facescape/run_pipeline.py), so verts/frame_ids each have a
    single entry (frame_ids = [0]) instead of being stacked across frames.

    wrapped_mesh.obj itself is in the shared origin space (see
    facescape_coordinate_transform.py's module docstring) that Phase 1 aligns
    every pair into before wrapping -- not this pair's own original FaceScape
    scan space. The verts are transformed out of origin space and back into
    that original scan space (solve_facescape_inverse_transform +
    apply_similarity_to_verts) before being saved, so the .pt this writes lines
    up with this pair's own un-transformed params.json/photos
    (facescape_raw_images_root), not the origin-space ones. A solve failure
    (missing/mismatched params.json) is exactly the "never raises" case below --
    nothing is written rather than silently exporting origin-space verts under
    a name that promises original-scan-space ones.

    Written to final_wrapped_data_root/facescape_<actor>/<expression>.pt --
    the "facescape_" actor-id prefix (not a separate subtree) keeps this
    sharing final_wrapped_data_root with Nersemble's <actor>/<sequence>.pt
    layout without any risk of actor-id collision (FaceScape actors are small
    ints like "1", which could otherwise collide with a Nersemble actor id).

    Never raises, same contract as export_wrapped_sequence()."""
    settings = get_settings()
    wrapped_mesh_path = settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj"

    if not wrapped_mesh_path.exists():
        log.warning(
            "export_wrapped_facescape_pair: no wrapped_mesh.obj found at %s for %s/%s",
            wrapped_mesh_path, actor_id, expression_id,
        )
        return None

    try:
        verts, faces = _load_obj_vertices_faces(wrapped_mesh_path)
    except (OSError, ValueError) as exc:
        log.warning("export_wrapped_facescape_pair: unreadable %s: %s", wrapped_mesh_path, exc)
        return None

    if facescape_invalid.is_invalid(actor_id, expression_id):
        # Invalid pair: run_pipeline_invalid.py already saved wrapped_mesh.obj
        # back in the pair's own raw (original scan) space -- nothing to undo.
        return _save_facescape_export(actor_id, expression_id, verts.astype(np.float32), faces, settings)

    solved = solve_facescape_inverse_transform(actor_id, expression_id)
    if solved is None:
        log.warning(
            "export_wrapped_facescape_pair: could not solve origin-space -> "
            "original-scan-space transform for %s/%s -- not exporting",
            actor_id, expression_id,
        )
        return None
    s, Rg, tg = solved
    original_space_verts = apply_similarity_to_verts(verts, s, Rg, tg).astype(np.float32)
    return _save_facescape_export(actor_id, expression_id, original_space_verts, faces, settings)


def _save_facescape_export(actor_id: str, expression_id: str, original_space_verts: np.ndarray, faces: np.ndarray, settings):
    """Writes final_wrapped_data_root/facescape_<actor>/<expression>.pt from
    verts already in the original scan space. Never raises."""
    verts_stack = torch.from_numpy(original_space_verts[None])  # (1, V, 3)
    faces_tensor = torch.from_numpy(faces)  # (F, 3)
    frame_ids_tensor = torch.tensor([0], dtype=torch.int64)

    output_path = settings.final_wrapped_data_root / f"facescape_{actor_id}" / f"{expression_id}.pt"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"faces": faces_tensor, "verts": verts_stack, "frame_ids": frame_ids_tensor},
            output_path,
        )
    except OSError as exc:
        log.warning("export_wrapped_facescape_pair: failed to write %s: %s", output_path, exc)
        return None

    log.info(
        "export_wrapped_facescape_pair: wrote %s (1 frame, %d verts, %d faces, original scan space)",
        output_path, verts_stack.shape[1], faces_tensor.shape[0],
    )
    return output_path


def delete_exported_facescape_pair(actor_id: str, expression_id: str) -> bool:
    """FaceScape counterpart to delete_exported_sequence() below -- same
    contract, just facescape_<actor>/<expression>.pt instead of
    <actor>/<sequence>.pt. Also best-effort removes a same-named .ply, a
    leftover artifact from export_wrapped_facescape_pair's old behavior
    (before it wrote original-scan-space verts directly into the .pt) --
    harmless no-op for any pair exported after that fix, which never creates
    one. Returns whether the .pt was removed."""
    settings = get_settings()
    pair_dir = settings.final_wrapped_data_root / f"facescape_{actor_id}"
    output_path = pair_dir / f"{expression_id}.pt"
    ply_path = pair_dir / f"{expression_id}.ply"

    try:
        ply_path.unlink()
        log.info("delete_exported_facescape_pair: removed legacy %s", ply_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("delete_exported_facescape_pair: failed to remove %s: %s", ply_path, exc)

    try:
        output_path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("delete_exported_facescape_pair: failed to remove %s: %s", output_path, exc)
        return False
    log.info("delete_exported_facescape_pair: removed %s", output_path)
    return True


def delete_exported_sequence(actor_id: str, sequence_id: str) -> bool:
    """Removes final_wrapped_data_root/<actor>/<sequence>.pt if it exists.

    Never raises, matching export_wrapped_sequence()'s contract -- this runs as
    a side effect of a successful /labels/unconfirm-confirmed, and a cleanup
    hiccup shouldn't undo a status change that already succeeded. Returns
    whether a file was actually removed (False if it was already absent)."""
    settings = get_settings()
    output_path = settings.final_wrapped_data_root / actor_id / f"{sequence_id}.pt"
    try:
        output_path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("delete_exported_sequence: failed to remove %s: %s", output_path, exc)
        return False
    log.info("delete_exported_sequence: removed %s", output_path)
    return True


def rename_wrapped_to_old_facescape_pair(actor_id: str, expression_id: str) -> bool:
    """FaceScape counterpart to rename_wrapped_to_old_sequence() below -- same
    contract, but FaceScape has exactly one wrapped_mesh.obj per pair (no
    per-frame subdirectories, see export_wrapped_facescape_pair() above).
    Overwrites any old_wrapped_mesh.obj already there from a previous
    unconfirm -- this only ever keeps the immediately-previous wrap, not a
    full history. Returns whether a file was actually renamed."""
    settings = get_settings()
    wrapped_path = settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj"
    old_path = wrapped_path.with_name("old_wrapped_mesh.obj")
    try:
        wrapped_path.rename(old_path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("rename_wrapped_to_old_facescape_pair: failed to rename %s: %s", wrapped_path, exc)
        return False
    log.info("rename_wrapped_to_old_facescape_pair: renamed %s -> %s", wrapped_path, old_path)
    return True


def rename_wrapped_to_old_sequence(actor_id: str, sequence_id: str) -> int:
    """Renames wrapped_mesh.obj -> old_wrapped_mesh.obj for every frame of this
    actor/sequence pair (Nersemble: one subdirectory per frame, unlike
    FaceScape's single file -- see rename_wrapped_to_old_facescape_pair()
    above). Never raises, matching export_wrapped_sequence()'s contract --
    this runs as a side effect of a successful /labels/unconfirm, and a
    rename hiccup shouldn't undo a status change that already succeeded.
    Returns how many frames were actually renamed."""
    settings = get_settings()
    sequence_output_dir = settings.output_root / actor_id / sequence_id
    try:
        frame_dirs = sorted(p for p in sequence_output_dir.iterdir() if p.is_dir() and p.name.isdigit())
    except OSError as exc:
        log.warning("rename_wrapped_to_old_sequence: cannot list %s: %s", sequence_output_dir, exc)
        return 0

    renamed = 0
    for frame_dir in frame_dirs:
        wrapped_path = frame_dir / "wrapped_mesh.obj"
        try:
            wrapped_path.rename(frame_dir / "old_wrapped_mesh.obj")
            renamed += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            log.warning("rename_wrapped_to_old_sequence: failed to rename %s: %s", wrapped_path, exc)
    log.info("rename_wrapped_to_old_sequence: renamed %d frame(s) for %s/%s", renamed, actor_id, sequence_id)
    return renamed
