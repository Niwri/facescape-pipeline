# Sapiens2: 2D Human Pose Estimation

308-keypoint top-down pose estimation including detailed face (274 keypoints),
hand, and foot keypoints. Predictions follow the [Sociopticon keypoint format](../sapiens/pose/configs/_base_/keypoints308.py).

## Model Zoo

Download checkpoints from [HuggingFace](https://huggingface.co/facebook/sapiens2)
and place them under `$SAPIENS_CHECKPOINT_ROOT` (default: `~/sapiens2_host`).

| Model | Checkpoint Path |
|-------|-----------------|
| Sapiens2-0.4B | `$SAPIENS_CHECKPOINT_ROOT/pose/sapiens2_0.4b_pose.safetensors` |
| Sapiens2-0.8B | `$SAPIENS_CHECKPOINT_ROOT/pose/sapiens2_0.8b_pose.safetensors` |
| Sapiens2-1B   | `$SAPIENS_CHECKPOINT_ROOT/pose/sapiens2_1b_pose.safetensors` |
| Sapiens2-5B   | `$SAPIENS_CHECKPOINT_ROOT/pose/sapiens2_5b_pose.safetensors` |

### Person Detector

```bash
hf download facebook/detr-resnet-101-dc5 \
    --local-dir "${SAPIENS_CHECKPOINT_ROOT}/detector/detr-resnet-101-dc5"
```

## Inference Guide

Runs on demo set (`demo/data`, 100 frames) by default:

```bash
cd $SAPIENS_ROOT/sapiens/pose
./scripts/demo/keypoints308.sh
```

Open the script and adjust:
- `INPUT` — path to your image directory (default: `../../demo/data`)
- `OUTPUT` — where to save visualizations
- `MODEL_NAME` — uncomment the model size you want to use
- `LINE_THICKNESS`, `RADIUS`, `KPT_THRES` — visualization style
- `JOBS_PER_GPU`, `GPU_IDS` — parallelism (defaults: 2 jobs/GPU on GPUs 0–7)

Outputs (visualization images + per-image keypoint JSONs) are written to `$OUTPUT`.

## Resources

- Demo: [facebook/sapiens2-pose](https://huggingface.co/spaces/facebook/sapiens2-pose)
- Models: [0.4B](https://huggingface.co/facebook/sapiens2-pose-0.4b), [0.8B](https://huggingface.co/facebook/sapiens2-pose-0.8b), [1B](https://huggingface.co/facebook/sapiens2-pose-1b), [5B](https://huggingface.co/facebook/sapiens2-pose-5b)
- Detector: [facebook/detr-resnet-101-dc5](https://huggingface.co/facebook/detr-resnet-101-dc5)
