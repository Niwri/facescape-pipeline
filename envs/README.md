# Python environments

The pipeline uses two venvs. The `*.freeze.txt` files are exact `pip freeze --all` exports of the
working venvs on Medusa (2026-10-12).

| venv | Python | Used for | `.env` key |
|---|---|---|---|
| `faceview_venv` | 3.10 | FaceView backend, propagation, wrapping, rendering | `VENV_PYTHON`, `VENV_ACTIVATE` |
| `faceview_sapiens_venv` | 3.12 | Sapiens landmark step | `SAPIENS_PYTHON` (backend + wrap_script `.env`) |

`FaceView/backend/requirements.txt` only lists the backend web stack (FastAPI etc.). Use the freeze
files for a full environment.

## Rebuild

Some packages are compiled against torch/CUDA, so install them in three steps:
1. torch (from the PyTorch wheel index);
2. everything that is a plain pip wheel;
3. the git/CUDA packages, built against the torch from step 1.

```bash
# --- faceview_venv (Python 3.10, CUDA 12.1 wheels on Medusa) ---
python3.10 -m venv faceview_venv && . faceview_venv/bin/activate
pip install -U pip setuptools wheel
# torch: Medusa has a pruned nightly (2.6.0.dev20241112+cu121); stable 2.6.0 is the closest replacement.
# Pick the CUDA tag (cu121/cu124/cu128 ...) that matches the new cluster's driver/GPU.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
grep -vE '^(torch|torchvision|torchaudio)==| @ git\+|^-e |^flash_attn' faceview_venv.freeze.txt > /tmp/fv_req.txt
pip install -r /tmp/fv_req.txt
# compiled against the torch above (needs nvcc matching torch's CUDA):
pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
  "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" \
  "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
# Optional: present in the Medusa venv but not imported by any code in this repo.
# pip install --no-build-isolation flash_attn==2.5.8 \
#   "git+https://github.com/ethz-vlg/pointcept.git@2082918692fef56f756c85cc067acdd44efb48ef#subdirectory=libs/pointops"

# --- faceview_sapiens_venv (Python 3.12, CUDA 12.6 wheels on Medusa) ---
python3.12 -m venv faceview_sapiens_venv && . faceview_sapiens_venv/bin/activate
pip install -U pip setuptools wheel
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
grep -vE '^(torch|torchvision|torchaudio)==| @ git\+|^-e ' faceview_sapiens_venv.freeze.txt > /tmp/sap_req.txt
pip install -r /tmp/sap_req.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
# sapiens: on Medusa it is an editable install of a private sapiens2 checkout (git+ssh, org account).
# Install the copy shipped in this repo instead:
pip install -e ../FaceView/sapiens2
```

## Notes

- **CoTracker** is not pip-installed. `run_neutral_cotracker_propagation.py` adds `FaceView/co-tracker` to
  `sys.path`. Its checkpoint is `FaceView/co-tracker/checkpoints/scaled_offline.pth`.
- **`TRACKER_PYTHON_BIN`** (backend MediaPipe/CoTracker subprocesses) points at another user's conda env
  on Medusa. `faceview_venv` has `mediapipe` installed, so pointing it at `faceview_venv/bin/python` is the
  simplest choice. This has not been tested.
- **GPU architecture.** These venvs target the Medusa GPUs. For Blackwell (sm_120) GPUs you need
  CUDA ≥ 12.8 torch wheels and rebuilt CUDA extensions. Medusa's sm_120 variants (`faceview_venv_sm120`,
  `faceview_sapiens_venv_sm120`) were built with scripts that are kept for reference at
  `/scratch/ondemand32/irwinngo/envs/build_faceview_*sm120*.sh`.
- **Exact pins.** Freeze pins are exact. If a version is no longer on PyPI, relax that pin rather than
  hunting for it.
