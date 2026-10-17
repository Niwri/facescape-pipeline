# Sapiens2: Pointmap (3D) Estimation

Per-pixel 3D point estimation — for each pixel, the model predicts its (x, y, z)
position in the camera coordinate frame. Together this forms a dense 3D point
cloud aligned with the input image.

## Model Zoo

Download checkpoints from [HuggingFace](https://huggingface.co/facebook/sapiens2)
and place them under `$SAPIENS_CHECKPOINT_ROOT` (default: `~/sapiens2_host`).

| Model | Checkpoint Path |
|-------|-----------------|
| Sapiens2-0.4B | `$SAPIENS_CHECKPOINT_ROOT/pointmap/sapiens2_0.4b_pointmap.safetensors` |
| Sapiens2-0.8B | `$SAPIENS_CHECKPOINT_ROOT/pointmap/sapiens2_0.8b_pointmap.safetensors` |
| Sapiens2-1B   | `$SAPIENS_CHECKPOINT_ROOT/pointmap/sapiens2_1b_pointmap.safetensors` |
| Sapiens2-5B   | `$SAPIENS_CHECKPOINT_ROOT/pointmap/sapiens2_5b_pointmap.safetensors` |

## Installation

Pointmap visualization requires `open3d`, which isn't installed by default:
```bash
pip install -e .[pointmap]
```

## Inference Guide

Runs on demo set (`demo/data`, 100 frames) by default:

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/demo/pointmap.sh
```

Open the script and adjust:
- `INPUT` — path to your image directory (default: `../../demo/data`)
- `OUTPUT` — where to save visualizations
- `MODEL_NAME` — uncomment the model size you want to use
- `JOBS_PER_GPU`, `GPU_IDS` — parallelism (defaults: 3 jobs/GPU on GPUs 0–7)

Outputs:
- 3D point cloud (`.ply` files) viewable in any Open3D / MeshLab / Blender viewer
- 2D depth visualization (`.jpg`)

## Resources

- Demo: [facebook/sapiens2-pointmap](https://huggingface.co/spaces/facebook/sapiens2-pointmap)
- Models: [0.4B](https://huggingface.co/facebook/sapiens2-pointmap-0.4b), [0.8B](https://huggingface.co/facebook/sapiens2-pointmap-0.8b), [1B](https://huggingface.co/facebook/sapiens2-pointmap-1b), [5B](https://huggingface.co/facebook/sapiens2-pointmap-5b)
