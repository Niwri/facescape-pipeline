# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sapiens.dense.models import init_model
from tqdm import tqdm


def load_mask_for_frame(seg_dir, frame_idx, frame_name, image_shape):
    """Load a per-frame mask for a video frame. Defaults to all-True if missing."""
    mask = np.ones(image_shape[:2], dtype=bool)
    if seg_dir is None:
        return mask

    frame_base = frame_name.rsplit(".", 1)[0]
    candidates = [
        os.path.join(seg_dir, f"{frame_base}.npy"),
        os.path.join(seg_dir, f"{frame_base}_seg.npy"),
        os.path.join(seg_dir, f"{frame_idx:06d}.npy"),
        os.path.join(seg_dir, f"{frame_idx:06d}_seg.npy"),
        os.path.join(seg_dir, frame_name),
    ]

    for mp in candidates:
        if not os.path.exists(mp):
            continue
        if mp.endswith("_seg.npy"):
            m = np.load(mp)  ## H x W, float; class labels
            mask = m > 0
        elif mp.endswith(".npy"):
            mask = np.load(mp)  ## H x W, boolean
        else:
            mask = cv2.imread(mp)[:, :, 0] > 0
        break

    return mask


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--seg_dir", "--seg-dir", default=None, help="Path to segmentation dir"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--no-black-background",
        "--no_black_background",
        action="store_true",
        help="No black background",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="If provided, do not save .npy prediction files",
    )
    parser.add_argument(
        "--input_path",
        "--input-path",
        default="/scratch/ondemand32/irwinngo/processed/017/EMO-1-shout+laugh/cam_222200038_delegation.mp4",
        help="Path to input video",
    )

    args = parser.parse_args()
    output_path = "/scratch/ondemand32/irwinngo/video_output"
    os.makedirs(output_path, exist_ok=True)
    config_path = "sapiens/dense/configs/normal/metasim_render_people/sapiens2_1b_normal_metasim_render_people-1024x768.py"
    checkpoint_path = "/scratch/ondemand32/irwinngo/models/sapiens2_host/normal/sapiens2_1b_normal.safetensors"
    input_path = args.input_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    video_stem = os.path.splitext(os.path.basename(input_path))[0]

    seg_dir = args.seg_dir

    # Defer model init until we actually need to run inference (skip-if-cached).
    model = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    frame_idx = 0
    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc="normal: render")
    while True:
        ret, image = cap.read()
        if not ret:
            break

        frame_name = f"{video_stem}_{frame_idx:06d}.png"
        base_path = os.path.join(output_path, f"{video_stem}_{frame_idx:06d}")
        normal_npy_path = f"{base_path}_normal.npy"

        mask = load_mask_for_frame(seg_dir, frame_idx, frame_name, image.shape)

        if os.path.exists(normal_npy_path):
            normal = np.load(normal_npy_path).astype(np.float32)
        else:
            if model is None:
                model = init_model(config_path, checkpoint_path, device=args.device)

            data = model.pipeline(dict(img=image))  ## resize and pad
            data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
            inputs, data_samples = data["inputs"], data["data_samples"]

            with torch.no_grad():
                normal = model(inputs)  # normal is 1 x 3 x H x W
                normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(
                    min=1e-8
                )

            pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"][
                "padding_size"
            ]
            normal = normal[
                :,
                :,
                pad_top : inputs.shape[2] - pad_bottom,
                pad_left : inputs.shape[3] - pad_right,
            ]

            normal = F.interpolate(
                normal,
                size=(image.shape[0], image.shape[1]),
                mode="bilinear",
                align_corners=False,
            )

            normal = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)  ## H x W x 3

            if not args.no_save_predictions:
                np.save(normal_npy_path, normal.astype(np.float16))

        normal_vis = normal.copy()
        normal_vis[mask == 0] = -1
        normal_vis = ((normal_vis + 1) / 2 * 255).astype(np.uint8)
        normal_vis = normal_vis[:, :, ::-1]

        if args.no_black_background:
            normal_vis[mask == 0] = image[mask == 0]

        vis_image = np.concatenate([image, normal_vis], axis=1)
        if writer is None:
            writer = cv2.VideoWriter(
                os.path.join(output_path, f"{video_stem}_normal_rendered.mp4"),
                fourcc,
                fps,
                (vis_image.shape[1], vis_image.shape[0]),
            )
        writer.write(vis_image)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()