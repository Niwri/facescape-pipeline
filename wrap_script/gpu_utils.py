"""GPU-memory-aware backpressure for wrap_script's heavy CUDA steps (Meshroom,
face_ray_masking's SAM segmentation, Wrap's compute step).

Under --parallel > 1 in run_pipeline_batch.py, several of these run concurrently
and share one GPU (see run_pipeline_batch.py's docs/history for the profiling that
motivated this: a single job's peak GPU usage was ~11.8GB against a 24GB card, most
of which was already occupied by an unrelated resident process). Rather than launch
a step and let it crash on CUDA OOM, wait_for_gpu_memory blocks until enough memory
is actually free, turning contention into a queue instead of a crash.

wait_for_gpu_memory is a context manager, not a plain call -- checking free memory
and then returning, on its own, isn't actually safe under concurrency: two separate
processes (each pair here is its own subprocess, see run_pipeline_batch.py) can
each poll torch.cuda.mem_get_info(), each see enough memory free, and each proceed
into e.g. Meshroom at the same moment -- Meshroom is an external subprocess whose
memory usage ramps up over several seconds rather than jumping to its peak
instantly, so the second caller's check can pass before the first caller's usage
has caught up to what it declared it needs. This was observed for real: --parallel
4 across 2 GPUs (2 pairs sharing each 24GB Titan) hit SIGKILL (-9, an OOM kill) on
several frames.

The fix is a per-GPU *semaphore*, not a per-GPU mutex: every active caller records
its own declared `required_mb` in a small shared reservations file
(gpu_<host>_<index>.reservations.json -- see the hostname note below), and a new
caller is only admitted once
(actual free memory) - (sum of every OTHER active caller's declared requirement)
covers its own required_mb (plus SAFETY_MARGIN_MB headroom). Treating every other
active reservation as fully "spoken for" -- even before that caller's usage has
actually ramped up -- closes the same race a plain memory check has (see above),
without needlessly serializing callers that together still fit: on a 48GB card,
an 11GB Meshroom job and a second, unrelated 11GB job can run at the same time,
where the original single-holder-at-a-time lock would have made the second job
wait out the first for no reason. Only genuinely competing requests (declared
requirements that don't all fit at once) end up serialized -- which, on the
24GB-card/~11.8GB-job case this module was originally built for, still means
"one at a time," exactly as before.

Reservation bookkeeping (read-check-admit-write, and later remove-on-release) is
itself protected by a brief flock on gpu_<host>_<index>.lock -- unlike the old
mutex design, this lock is only ever held for that short read/modify/write, never
for the duration of the actual GPU step.

Lock/reservation files are scoped by *hostname*, not just CUDA index: GPU_LOCK_DIR
lives on an NFS mount shared across multiple clusters/machines, and "GPU 0" on one
host is a physically different, unrelated card from "GPU 0" on another -- indexing
by CUDA index alone would make two unrelated physical GPUs on different hosts
contend over (and count each other's reservations against) the same file. Because
of this, every entry actually present in a given gpu_<host>_<index>.reservations.json
was necessarily written by a process on that same host.

Stale reservations (owner process crashed/killed without reaching the `finally`
release, so its entry was never removed) are dropped automatically on every
admission check: an entry whose declared process is confirmed gone (checked via
os.kill(pid, 0), always meaningful here since every entry belongs to this file's
own host) or whose age exceeds MAX_RESERVATION_AGE_SECONDS (a liveness-check-
independent fallback) is treated as gone.
"""
from __future__ import annotations

import nersemble_paths
import contextlib
import fcntl
import json
import os
import socket
import time
import uuid
from pathlib import Path

import torch

GPU_LOCK_DIR = nersemble_paths.GPU_LOCKS_ROOT

# A reservation older than this is dropped on the next admission check
# regardless of host -- covers a crashed/killed remote-host process whose PID
# liveness we have no way to check locally. Generously above any single
# step's realistic runtime (the timeout below already bounds a *caller's* own
# patience; this bounds how long its *reservation* can outlive it).
MAX_RESERVATION_AGE_SECONDS = 3600.0

# Extra headroom kept unreserved on top of every admitted caller's declared
# requirement -- a step's actual peak can overshoot its own REQUIRED_MB
# constant slightly, and this absorbs that without needing each constant
# tuned to the exact byte.
SAFETY_MARGIN_MB = 500.0


def _physical_gpu_index() -> str:
    """The physical GPU id this process is actually bound to. Every subprocess
    sees its assigned GPU as device 0 locally (via CUDA_VISIBLE_DEVICES, see
    run_pipeline_batch.py's own per-pair GPU pinning), so torch.cuda.current_device()
    alone can't distinguish which physical card different concurrent processes
    are actually on -- CUDA_VISIBLE_DEVICES itself is the ground truth."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first:
            return first
    return "0"


def _read_reservations(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_reservations(state_path: Path, reservations: dict) -> None:
    state_path.write_text(json.dumps(reservations))


def _is_stale(entry: dict, now: float) -> bool:
    if now - entry.get("started", 0.0) > MAX_RESERVATION_AGE_SECONDS:
        return True
    if entry.get("host") == socket.gethostname():
        try:
            os.kill(entry["pid"], 0)
        except ProcessLookupError:
            return True  # no such process on this host anymore -- truly gone
        except PermissionError:
            pass  # exists, just owned by someone else -- still alive
    return False


@contextlib.contextmanager
def wait_for_gpu_memory(
    required_mb: float,
    label: str = "",
    poll_interval: float = 10.0,
    timeout: float | None = 1800.0,
):
    """Context manager: blocks until admission is granted -- i.e. until
    (actual free GPU memory) - (every other active caller's declared
    required_mb) covers this caller's own required_mb + SAFETY_MARGIN_MB --
    then holds that reservation for the duration of the `with` block and
    releases it on exit (success or exception). No-op (nullcontext) if CUDA
    isn't available. Raises TimeoutError if `timeout` seconds pass without
    being admitted (pass None to wait indefinitely).

    Usage:
        with wait_for_gpu_memory(MESHROOM_REQUIRED_MB, label="Meshroom reconstruction"):
            target_path = meshroom_compute(...)
    """
    if not torch.cuda.is_available():
        yield
        return

    gpu_index = _physical_gpu_index()
    host = socket.gethostname()
    GPU_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    # Scoped by hostname, not just GPU index: this lock directory is on an
    # NFS mount shared across multiple clusters/machines, and "GPU 0" on one
    # host is a physically different, unrelated card from "GPU 0" on another
    # -- indexing the lock/reservation files by CUDA index alone would make
    # two unrelated GPUs on different hosts contend over the same file.
    lock_path = GPU_LOCK_DIR / f"gpu_{host}_{gpu_index}.lock"
    state_path = GPU_LOCK_DIR / f"gpu_{host}_{gpu_index}.reservations.json"
    reservation_id = f"{host}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    start = time.time()

    def _check_timeout(waiting_for: str) -> None:
        elapsed = time.time() - start
        if timeout is not None and elapsed > timeout:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for {waiting_for}"
                f"{f' before {label}' if label else ''} on GPU {gpu_index}."
            )

    while True:
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                now = time.time()
                reservations = {
                    rid: entry for rid, entry in _read_reservations(state_path).items()
                    if not _is_stale(entry, now)
                }
                other_reserved_mb = sum(entry["mb"] for entry in reservations.values())
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 ** 2)
                effective_free_mb = free_mb - other_reserved_mb

                admitted = effective_free_mb >= required_mb + SAFETY_MARGIN_MB
                if admitted:
                    reservations[reservation_id] = {
                        "host": socket.gethostname(), "pid": os.getpid(),
                        "mb": required_mb, "label": label, "started": now,
                    }
                _write_reservations(state_path, reservations)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        if admitted:
            break
        _check_timeout(f"{required_mb:.0f} MiB free GPU memory")
        print(f"[gpu_utils] Waiting for GPU {gpu_index} memory{f' before {label}' if label else ''}: "
              f"need {required_mb:.0f} (+{SAFETY_MARGIN_MB:.0f} margin) MiB, "
              f"{effective_free_mb:.0f} MiB effectively free ({free_mb:.0f} MiB actually free, "
              f"{other_reserved_mb:.0f} MiB reserved by {len(reservations)} other caller(s)), "
              f"waited {time.time() - start:.0f}s, retrying in {poll_interval:.0f}s...")
        time.sleep(poll_interval)

    try:
        yield
    finally:
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                reservations = _read_reservations(state_path)
                reservations.pop(reservation_id, None)
                _write_reservations(state_path, reservations)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
