# FaceView

A web-based human-in-the-loop tool for correcting multiview facial landmarks. A React/Vite
frontend plays synchronized multi-camera video with a landmark overlay; a FastAPI backend
serves the dataset, stores landmark edits, and re-propagates a correction to later frames
with CoTracker v3.

This doc covers two things:

1. **[Collaborator setup](#collaborator-setup)** — what someone helping you label needs to
   install and configure to run the frontend + backend on their own machine.
2. **[Sharing the data](#sharing-the-data)** — how to get them the `data/`, `processed/`,
   and `landmarks/` folders the app reads from.

---

## Architecture

| Piece | What it needs | Approx. size |
|---|---|---|
| `data/<actor>/` | Raw per-camera `.mp4`s + `calibration/camera_params.json` | ~30GB/actor (353GB total today) |
| `processed/<actor>/<sequence>/` | Small `*_delegation.mp4` files the frontend actually plays | 5.2GB total |
| `landmarks/<actor>/<sequence>/<camera>/frame_*.json` | The actual annotation data — this is what gets edited | 15GB total |

The frontend never talks to `data/` directly — it only plays `processed/` delegation videos
and reads/writes `landmarks/` JSON through the backend's `/api` routes. `data/` is read
**server-side**, by the backend, when a landmark edit triggers CoTracker propagation (it
crops and tracks on the raw-resolution video, not the delegation video). So a collaborator
needs `data/` too, just not necessarily for every actor — only the ones they're labeling.

---

## Collaborator setup

### 1. Get the code

```bash
git clone git@github.com:Niwri/FaceView.git
cd FaceView
```

**Heads up:** `co-tracker/`, `sapiens2/`, and `nersemble-data/` show up in `git ls-files` as
submodule-style entries, but this repo has no `.gitmodules` file, so a fresh clone leaves
them as **empty directories** — git has no URL to fetch them from. You only need one of the
three for day-to-day labeling:

- **`co-tracker/`** — required. It's Meta's [CoTracker](https://github.com/facebookresearch/co-tracker),
  imported directly by the backend to run propagation. Clone it into the repo root:
  ```bash
  git clone https://github.com/facebookresearch/co-tracker.git
  ```
  Then get the checkpoint used by this app (`co-tracker/checkpoints/scaled_offline.pth`,
  ~97MB) — ask whoever's coordinating (it's not in git) and drop it at that exact path.

- **`sapiens2/`** and **`nersemble-data/`** — only used by preprocessing scripts
  (`scripts/preprocess/process_sequence_sapiens.py` and friends) that generate the initial
  landmark guesses for a brand-new sequence. If you're just correcting landmarks on
  sequences that already have `landmarks/` data, **you don't need these** — skip them.

You'll also need two small config files that aren't tracked in git — ask for copies and
place them at:
- `default_cameras.json` (repo root)
- `configs/face_indices.json` (optional — the app degrades gracefully if it's missing)

### 2. Python environment

The reference environment this app was built/run against:
- **Python 3.12.9**
- **PyTorch 2.7.1**, built for **CUDA 12.6** (`torch==2.7.1+cu126`)
- NVIDIA driver 550.67 (CUDA 12.4 max) — match torch's CUDA build to *your own* GPU/driver,
  it doesn't need to be identical, just compatible. If you don't have a GPU, a CPU build
  works too, just makes propagation noticeably slower.

Set up a virtualenv and install:

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install -r backend/requirements.txt

# Install torch matching your CUDA version — check https://pytorch.org/get-started/locally/
# for the right --index-url. Example, matching the reference env (CUDA 12.6):
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

`backend/requirements.txt` covers FastAPI/uvicorn/websockets/pydantic-settings; CoTracker
itself (`co-tracker/setup.py`) declares no extra dependencies beyond torch/numpy.

### 3. Path configuration

The backend reads its paths from environment variables (via `pydantic-settings`, loaded
from a `.env` file in `backend/`). Create `backend/.env`:

```bash
DATA_ROOT=/path/to/your/local/data
PROCESSED_ROOT=/path/to/your/local/processed
LANDMARKS_ROOT=/path/to/your/local/landmarks
```

These map directly to `Settings` fields in `backend/app/core/config.py` (env vars are
matched case-insensitively). Everything else in that file has a working default:

| Setting | Default | When you'd change it |
|---|---|---|
| `configs_root` | `configs` (relative to cwd) | if you placed `face_indices.json` elsewhere |
| `default_cameras_path` | `<repo root>/default_cameras.json` | if you placed it elsewhere |
| `tracker_python_bin` | a path on Irwin's cluster | only needed for `POST .../preprocess` (fresh sequences) — irrelevant if you're only correcting existing landmarks |
| `propagate_window` | `48` | how many frames CoTracker re-tracks after an edit |

The three directories (`DATA_ROOT`, `PROCESSED_ROOT`, `LANDMARKS_ROOT`) are auto-created on
startup if missing, but for existing actors you'll want them populated — see
[Sharing the data](#sharing-the-data) below.

### 4. Frontend setup

Reference environment: **Node.js v20.19.5** (npm 10.8.2), matching Vite 5's own requirement
(`^18.0.0 || >=20.0.0`). `package.json` has no `engines` field, so nothing enforces this —
if you're on Node 18+ it'll almost certainly work, but if you hit odd Vite/TypeScript errors,
check your Node version first. Using [nvm](https://github.com/nvm-sh/nvm) is the easiest way
to match it exactly:

```bash
nvm install 20.19.5
nvm use 20.19.5
```

```bash
cd frontend
npm install
npm run dev
```

This starts Vite on `http://localhost:5173`. By default it proxies `/api`, `/ws`, and
`/processed` to `http://localhost:8000` (see `frontend/vite.config.ts`) — fine if backend
and frontend run on the same machine. If the backend runs elsewhere, either:
- edit the `proxy` targets in `vite.config.ts`, or
- set `VITE_API_BASE` / `VITE_MEDIA_BASE` env vars (see `frontend/src/config.ts`) to the
  backend's full URL — the backend already allows all CORS origins, so this just works.

When SSHing, use the -X -L flags for portforwarding to access the website remotely. (i.e, `ssh username@medusa.dgp.toronto.edu -X -L 5173:127.0.0.1:5173`, then `ssh odysseus.dgpsrv.sandbox -X -L 5173:127.0.0.1:5173`))

### 5. Running the backend

```bash
cd backend
source ../venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` matters if the frontend (or teammates) will reach this backend over the
network rather than from `localhost`. CoTracker is imported directly (not a subprocess), so
just having `co-tracker/` present at the repo root (alongside `backend/`) is enough — no
extra `PYTHONPATH` setup needed, `loop_service.py` adds it automatically.

---

## Sharing the data

Given the sizes involved (**353GB** `data/`, 5.2GB `processed/`, 15GB `landmarks/`), don't
just hand over the whole tree — most of `data/` is raw video for actors nobody's actively
labeling.

### Recommended: rsync over SSH (same network/cluster)

If your collaborator can SSH into the machine holding the data:

```bash
# processed/ and landmarks/ are small enough to share in full
rsync -avP /scratch/ondemand32/irwinngo/processed/ collaborator@host:/path/to/FaceView/processed/
rsync -avP /scratch/ondemand32/irwinngo/landmarks/  collaborator@host:/path/to/FaceView/landmarks/

# data/ — only send the actor(s) they're actually assigned, e.g. actor 017:
rsync -avP /scratch/thirty/irwinngo/data/017/ collaborator@host:/path/to/FaceView/data/017/
```

(Or run these the other direction, pulling from their end, whichever side has the open SSH
port.)

### If they're on a different network entirely

- **Cloud bucket / rclone**: push `processed/` + `landmarks/` (and the specific `data/<actor>/`
  folders needed) to S3/GCS/a shared Google Drive with `rclone`, then have them pull down.
- **tar + split**, for raw file transfer without cloud storage:
  ```bash
  tar czf - data/017 | split -b 5G - data_017.tar.gz.part_
  # transfer the parts, then on the other end:
  cat data_017.tar.gz.part_* | tar xzf -
  ```

### Syncing `landmarks/` back (this is the part that actually changes)

`landmarks/` is the one folder that's actively edited on both sides, and there's no shared
database behind it — it's just JSON files on each person's disk. To avoid clobbering each
other's edits:
- **Split by actor/sequence.** Assign each collaborator a disjoint set of
  actor/sequence pairs so nobody's editing the same `landmarks/<actor>/<sequence>/` tree at
  once.
- Periodically `rsync -avP` their `landmarks/<actor>/` back to the canonical copy (whoever's
  treated as "source of truth"), one actor directory at a time, so it's obvious what's being
  merged in.

---

## Notes

- `whitelist.json` (in `wrap_script/`, separate from this repo) lists the landmark indices
  that are actually hand-labeled/corrected — everything else is either derived (FLAME/sapiens)
  or intentionally unlabeled.
- `camera_render_landmarks.py` renders a visualization of labeled landmarks across multiple
  cameras concatenated horizontally, for spot-checking a sequence outside the web UI.
- `scripts/preprocess/process_sequence_sapiens.py <ACTOR_ID> <SEQUENCE_ID> <cameras>` is
  what generates the *initial* landmark guesses for a new sequence (FLAME-projected base,
  sapiens2-overridden eyelid/lip points) — not part of the day-to-day correction workflow.
