# Local labeling (Nersemble + Ava-256)

Label on your PC against local copies of the overlay videos, then merge the
labels back into the host's `label_tracker.json`.

1. **Host:** snapshot the labels (from `FaceView/`):

       python scripts/local_labeling/export_local_bundle.py --out /scratch/ondemand32/irwinngo/local_bundle

   It prints the `scp`/`rsync` commands for the bundle and the three video
   folders (`line_renders/`, `point_overlay_renders/`, `line_renders_ava256/`).

2. **PC:** copy `local_labeler.py`, the bundle and the videos, then run it.
   It needs Python 3.9+ and no extra packages:

       python local_labeler.py --bundle ./local_bundle --videos ./VIDEOS

   Open http://localhost:8765. Every click saves to `local_bundle/label_tracker.json`.

3. **Host:** copy that `label_tracker.json` back (e.g. to `irwinngo/label_tracker.local.json`) and merge:

       # dry run: shows what would change
       python scripts/local_labeling/merge_local_labels.py /scratch/ondemand32/irwinngo/label_tracker.local.json \
           --bundle /scratch/ondemand32/irwinngo/local_bundle
       # then add --apply to do it (it backs up first)

   Conflicts are pairs or captures that also changed on the host since step 1.
   They are skipped and listed; add `--prefer-local` to force your labels.
   Keep the bundle folder on the host until the merge is done: it is the merge base.

Do **not** copy the edited file straight over the real `label_tracker.json`.
The merge replays each change through FaceView's own functions, so the
`.pt` exports, `wrapped_mesh.obj` renames and Ava-256 archiving happen exactly
as they would in the web app.
