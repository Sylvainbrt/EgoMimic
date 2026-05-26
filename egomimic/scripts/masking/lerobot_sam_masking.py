#!/usr/bin/env python3
"""
Apply SAM2-based hand masking to a converted LeRobot → EgoMimic HDF5 file.

Run this AFTER convert_lerobot_to_egomimic.py.

Usage:
    python lerobot_sam_masking.py \
        --dataset /data/pick_sponge_robot.hdf5 \
        --sam \
        [--debug]
"""

import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

from egomimic.utils.egomimicUtils import ARIA_INTRINSICS, EXTRINSICS
from egomimic.scripts.masking.utils import SAM


def _resolve_extrinsics(extrinsics_key: str, arm: str):
    if extrinsics_key not in EXTRINSICS:
        available = ", ".join(sorted(EXTRINSICS.keys()))
        raise KeyError(f"Unknown extrinsics key '{extrinsics_key}'. Available keys: {available}")

    resolved = EXTRINSICS[extrinsics_key]
    if isinstance(resolved, dict):
        if arm not in resolved:
            available_arms = ", ".join(sorted(resolved.keys()))
            raise KeyError(
                f"Extrinsics key '{extrinsics_key}' does not define arm '{arm}'. "
                f"Available arms: {available_arms}"
            )
        return {arm: resolved[arm]}

    # Backward compatibility for older single-matrix entries such as ariaJul29L/R.
    return {arm: resolved}


def sam_processing(
    dataset: str,
    extrinsics_key: str,
    arm: str,
    debug: bool = False,
    output_dataset: str | None = None,
    write_masked: bool = True,
    start_episode: int = 0,
    max_episodes: int | None = None,
):
    """
    Iterate over all demos in the EgoMimic HDF5 and apply SAM2 hand masking.
    Writes the following datasets into each demo/obs group:
        - front_img_1_masked  (T, H, W, 3)
        - front_img_1_mask    (T, H, W)  bool
        - front_img_1_line    (T, H, W, 3)

    Args:
        dataset: path to the EgoMimic HDF5 file
        extrinsics_key: key inside egomimicUtils.EXTRINSICS
        arm: which robot arm to project
        output_dataset: optional destination path. If provided and different from
            dataset, the input HDF5 is copied there first and masking is applied
            to the copy.
        write_masked: whether to write the full front_img_1_masked dataset. Set
            False to reduce disk writes when only the line overlay is needed.
        start_episode: zero-based demo index to start from
        max_episodes: optional number of demos to process from start_episode
        debug:   enable debug visualisations inside SAM
    """
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()
    robot_extrinsics = _resolve_extrinsics(extrinsics_key, arm)
    print(f"Using extrinsics='{extrinsics_key}' arm='{arm}'")

    dataset_path = Path(dataset)
    target_path = Path(output_dataset) if output_dataset is not None else dataset_path

    if target_path != dataset_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying source dataset to: {target_path}")
        shutil.copy2(dataset_path, target_path)

    print(f"Writing results to: {target_path}")

    with h5py.File(target_path, "r+") as data:
        demo_keys = sorted(data["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if start_episode < 0:
            raise ValueError(f"start_episode must be >= 0, got {start_episode}")
        demo_keys = demo_keys[start_episode:]
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError(f"max_episodes must be > 0, got {max_episodes}")
            demo_keys = demo_keys[:max_episodes]
        print(f"Processing {len(demo_keys)} episode(s) starting from demo_{start_episode}")

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
                    imgs, qpos, robot_extrinsics, ARIA_INTRINSICS, arm=arm, debug=debug
                )

                

                keys_to_delete = ["front_img_1_mask", "front_img_1_line"]
                if write_masked:
                    keys_to_delete.append("front_img_1_masked")
                for key in keys_to_delete:
                    if key in obs:
                        print(f"  [{demo_key}] Deleting existing '{key}'")
                        del obs[key]

                T, H, W, _ = imgs.shape
                if write_masked:
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
                data.flush()


def main(args):
    if args.sam:
        sam_processing(
            args.dataset,
            extrinsics_key=args.extrinsics_key,
            arm=args.arm,
            debug=args.debug,
            output_dataset=args.output_dataset,
            write_masked=not args.line_only,
            start_episode=args.start_episode,
            max_episodes=args.max_episodes,
        )


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
    parser.add_argument(
        "--extrinsics-key", type=str, default="ariaSylvain",
        help="Key from egomimicUtils.EXTRINSICS to use for projection.",
    )
    parser.add_argument(
        "--arm", type=str, choices=["left", "right"], default="left",
        help="Which single-arm configuration this dataset uses.",
    )
    parser.add_argument(
        "--output-dataset", type=str, default=None,
        help="Optional destination HDF5 path. If set, copy the input file there and write masking results to the copy.",
    )
    parser.add_argument(
        "--line-only", action="store_true",
        help="Only write front_img_1_line and skip front_img_1_masked to reduce disk usage.",
    )
    parser.add_argument(
        "--start-episode", type=int, default=0,
        help="Zero-based demo index to start from, e.g. 0 for demo_0.",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Optional number of demos to process from --start-episode.",
    )

    args = parser.parse_args()

    main(args)
