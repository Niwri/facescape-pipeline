# Sapiens2: Human Image Matting

Per-pixel alpha matting for human subjects. Predicts a soft foreground mask
(`alpha` in `[0, 1]`) plus a pre-multiplied foreground RGB output.

<p align="center">
  <img src="assets/sapiens2_1b_matting_demo.gif" alt="Sapiens2-1B matting demo" title="Sapiens2-1B matting demo" width="960"/>
</p>

## Inference

Runs on demo set (`demo/data`, 100 frames) by default:

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/demo/matting.sh
```

Open the script and adjust:
- `INPUT` — path to your image directory (default: `../../demo/data`)
- `OUTPUT` — where to save visualizations
- `MODEL_NAME` — model size (default: `sapiens2_1b`)
- `JOBS_PER_GPU`, `GPU_IDS` — parallelism (defaults: 3 jobs/GPU on GPUs 0–7)

Outputs per image:
- Side-by-side visualization: `[input | alpha matte | foreground on chroma green]`
- `--save_pred` (on by default in the script) additionally writes `<name>_alpha.npy`
  with the raw alpha as `float32` in `[0, 1]`.

## Resources

- Demo: [facebook/sapiens2-matting](https://huggingface.co/spaces/facebook/sapiens2-matting)
- Model: [1B](https://huggingface.co/facebook/sapiens2-matting-1b)
