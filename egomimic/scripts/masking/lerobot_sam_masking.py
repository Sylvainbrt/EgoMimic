#!/usr/bin/env python3
"""
Apply SAM2-based hand masking to a converted LeRobot → EgoMimic HDF5 file.

Run this AFTER convert_lerobot_to_egomimic.py.

Usage:
    python lerobot_sam_masking.py \
        --dataset /data/pick_sponge_robot.hdf5 \
        --sam \
        [--inpaint] \
        [--debug]
"""

import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm

from egomimic.utils.egomimicUtils import ARIA_INTRINSICS, EXTRINSICS
# Use ariaJul29 extrinsics to test, to be calibrated later
ROBOT_EXTRINSICS = EXTRINSICS["ariaJul29"]
from egomimic.scripts.masking.utils import SAM


def sam_processing(dataset: str, debug: bool = False):
    """
    Iterate over all demos in the EgoMimic HDF5 and apply SAM2 hand masking.
    Writes the following datasets into each demo/obs group:
        - front_img_1_masked  (T, H, W, 3)
        - front_img_1_mask    (T, H, W)  bool
        - front_img_1_line    (T, H, W, 3)

    Args:
        dataset: path to the EgoMimic HDF5 file
        debug:   enable debug visualisations inside SAM
    """
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    with h5py.File(dataset, "r+") as data:
        demo_keys = sorted(data["data"].keys(), key=lambda k: int(k.split("_")[1]))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for demo_key in tqdm(demo_keys, desc="SAM masking"):
                demo = data[f"data/{demo_key}"]

                if "obs/front_img_1" not in demo:
                    print(f"  [{demo_key}] No front_img_1 found, skipping.")
                    continue

                if "obs/ee_pose" not in demo:
                    print(f"  [{demo_key}] No ee_pose found, skipping.")
                    continue

                obs = demo["obs"]

                imgs       = obs["front_img_1"][:]        # (T, H, W, 3)
                qpos       = obs["joint_positions"][:]    # (T, 7)

                mask_images, line_images = sam.get_robot_mask_line_batched_from_qpos(
                    imgs, qpos, ROBOT_EXTRINSICS, ARIA_INTRINSICS, arm="right"
                )

                

                for key in ("front_img_1_masked", "front_img_1_mask", "front_img_1_line"):
                    if key in obs:
                        print(f"  [{demo_key}] Deleting existing '{key}'")
                        del obs[key]

                T, H, W, _ = imgs.shape
                obs.create_dataset(
                    "front_img_1_masked",
                    data=mask_images,
                    chunks=(1, H, W, 3),
                )
                obs.create_dataset(
                    "front_img_1_line",
                    data=line_images,
                    chunks=(1, H, W, 3),
                )


def main(args):
    if args.sam:
        sam_processing(args.dataset, debug=args.debug)


if __name__ == "__main__":
    """
    Usage:
        python lerobot_sam_masking.py \
            --dataset /data/pick_sponge_robot.hdf5 \
            --sam \
            [--debug]
    """
    parser = argparse.ArgumentParser(
        description="Apply SAM2 hand masking to a converted LeRobot EgoMimic HDF5."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the EgoMimic HDF5 file (output of convert_lerobot_to_egomimic.py).",
    )
    parser.add_argument(
        "--sam", action="store_true",
        help="Run SAM2 masking.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug visualisations.",
    )

    args = parser.parse_args()

    if not args.sam and not args.inpaint:
        parser.error("Specify at least one of --sam or --inpaint.")

    main(args)