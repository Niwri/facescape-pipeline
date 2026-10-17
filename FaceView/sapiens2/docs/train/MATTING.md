# Training Sapiens2 for Human Image Matting

In-the-wild matting on a mixture of relit / synthetic / captured datasets,
loaded via the `MattingBaseDataset` (annotation-driven JSON manifest) and
`MattingGSSDataset` (RGBA image manifest).

The dataset paths and `pretrained_checkpoint` in
`sapiens/dense/configs/matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py`
point to the internal locations used to train the released model — edit them
to point at your own data and backbone checkpoint before launching training.

## Launch

Single-node multi-GPU training:

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/matting/train/sapiens2_1b/node.sh
```

Open the script and adjust:
- `DEVICES` — GPU IDs (default: `0,1,2,3,4,5,6,7`)
- `TRAIN_BATCH_SIZE_PER_GPU` — per-GPU batch size (default: 8)
- `mode` — set to `debug` for a single-process run

Outputs are written under `Outputs/matting/train/<MODEL>/node/<timestamp>/`.

## Dataset Format

`MattingBaseDataset` expects an annotation JSON of the form:

```json
[
  {"image": "relpath/to/img.png", "mask": "relpath/to/mask.png"},
  ...
]
```

Where `mask` is either:
- a single-channel alpha image, or
- an RGBA image whose alpha channel is the matte and RGB channels are the
  foreground color before alpha pre-multiplication.

`MattingGSSDataset` expects a `.txt` manifest with one RGBA image path per line.
