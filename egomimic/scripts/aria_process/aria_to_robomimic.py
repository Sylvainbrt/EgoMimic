import os
import h5py

# mps_sample_path = "/coc/flash9/skareer6/Projects/EgoPlay/aria/aria_demo/simar/"

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from projectaria_tools.core.stream_id import StreamId
import numpy as np
import torch
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional

from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from aria_utils import (
    build_camera_matrix,
    undistort_to_linear,
    split_train_val_from_hdf5,
    slam_to_rgb,
)

import argparse

import json

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    WIDE_LENS_HAND_LEFT_K,
    ARIA_INTRINSICS,
    ARIA_WIDE_INTRINSICS,
    interpolate_keys,
    interpolate_arr
)
from egomimic.scripts.masking.utils import *

HORIZON = 10
STEP = 3


"""
Example usage
python aria_to_robomimic.py --dataset /coc/flash7/datasets/egoplay/oboo_aria_apr16/oboo_aria_apr16/ --out /coc/flash7/datasets/egoplay/oboo_aria_apr16/converted/oboo_aria_apr16_rightMimicplay.hdf5 --hand right
"""

# Load the VRS file


def single_file_conversion(dataset, mps_sample_path, filename, hand):
    """
    dataset: path to the dataset
    mps_sample_path: path to the mps sample
    filename: name of the vrs file
    hand: left, right, bimanual

    Returns: actions [N, HORIZON, ac_dim], front_img_1 [N, H, W, 3], ee_pose [N, ac_dim]
    """
    vrsfile = os.path.join(dataset, filename)

    # Hand tracking CSV (your recreated file)
    wrist_and_palm_poses_path = os.path.join(
        mps_sample_path, "hand_tracking", "wrist_and_palm_poses.csv"
    )

    provider = data_provider.create_vrs_data_provider(vrsfile)

    # Load hand tracking
    _ = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)

    device_calibration = provider.get_device_calibration()
    time_domain: TimeDomain = TimeDomain.DEVICE_TIME
    time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST



    stream_ids: Dict[str, StreamId] = {
        "rgb": StreamId("214-1"),
        "slam-left": StreamId("1201-1"),
        "slam-right": StreamId("1201-2"),
    }
    stream_labels: Dict[str, str] = {
        key: provider.get_label_from_stream_id(stream_id)
        for key, stream_id in stream_ids.items()
    }
    stream_timestamps_ns: Dict[str, List[int]] = {
        key: provider.get_timestamps_ns(stream_id, time_domain)
        for key, stream_id in stream_ids.items()
    }

    # Basic sanity
    if len(stream_timestamps_ns["rgb"]) == 0:
        return np.zeros((0, HORIZON, 3 if hand != "bimanual" else 6)), \
               np.zeros((0, 1, 1, 3), dtype=np.uint8), \
               np.zeros((0, 3 if hand != "bimanual" else 6))

    vrs_data_provider = data_provider.create_vrs_data_provider(vrsfile)

    mps_data_paths_provider = mps.MpsDataPathsProvider(mps_sample_path)
    mps_data_paths = mps_data_paths_provider.get_data_paths()
    mps_data_provider = mps.MpsDataProvider(mps_data_paths)

    transform = slam_to_rgb(vrs_data_provider)

    # Extract T_device_rgb_camera outside your loop
    device_calibration = vrs_data_provider.get_device_calibration()
    rgb_calib = device_calibration.get_camera_calib(stream_labels["rgb"])
    T_device_rgb_camera = rgb_calib.get_transform_device_camera()
    T_rgb_camera_device = T_device_rgb_camera.inverse()

    frame_length = len(stream_timestamps_ns["rgb"])
    print(f"Total RGB frames: {frame_length}")

    actions_list = []
    imgs_list = []
    ee_pose_list = []

    ac_dim = 3 if hand != "bimanual" else 6

    center_px_test = ARIA_INTRINSICS @ np.array([0, 0, 1, 1])
    print(f"Optical center should be at: {center_px_test[:2]}")

    for t in range(frame_length):
        if (t % 1000) == 0:
            print(f"{t} frames ingested")

        sample_timestamp_ns_t = stream_timestamps_ns["rgb"][t]

        # Get wrist pose at reference time
        wrist_t = mps_data_provider.get_wrist_and_palm_pose(
            sample_timestamp_ns_t, time_query_closest
        )
        if wrist_t is None:
            continue

        # Require chosen hand at reference time
        if hand == "right":
            if wrist_t.right_hand is None:
                continue
        elif hand == "left":
            if wrist_t.left_hand is None:
                continue
        else:  # bimanual: at least one hand at t
            if (wrist_t.left_hand is None) and (wrist_t.right_hand is None):
                continue

        # Load RGB frame; skip only this t if it fails
        try:
            frame_rgb = provider.get_image_data_by_time_ns(
                stream_ids["rgb"],
                sample_timestamp_ns_t,
                time_domain,
                time_query_closest,
            )[0]
        except Exception:
            continue

        img_t = undistort_to_linear(
            provider,
            stream_ids,
            raw_image=frame_rgb.to_numpy_array(),
        )

        # Closed-loop pose at t (camera pose); if missing we cannot define camera frame
        pose_t = mps_data_provider.get_closed_loop_pose(
            sample_timestamp_ns_t, time_query_closest
        )
        if pose_t is None:
            continue

        camera_matrix_t = build_camera_matrix(vrs_data_provider, pose_t)
        camera_t_inv = np.linalg.inv(camera_matrix_t)

        # Rotation to EgoMimic convention
        rotation_matrix = np.array([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]])



        # ee_pose at reference frame in camera_t
        if hand == "right":
            palm_dev = wrist_t.right_hand.palm_position_device
            # Ensure the evaluated 3D point strictly has shape (3,)
            palm_cam_t = np.array(T_rgb_camera_device @ palm_dev).flatten()

            # Rotate to EgoMimic convention
            ee_pose_obs_t = rotation_matrix @ palm_cam_t
        elif hand == "left":
            palm_dev = wrist_t.left_hand.palm_position_device
            palm_cam_t = np.array(T_rgb_camera_device @ palm_dev).flatten()
            ee_pose_obs_t = rotation_matrix @ palm_cam_t
        else:
            ee_pose_obs_t = np.zeros(6, dtype=np.float32)
            if wrist_t.left_hand is not None:
                palm_l_dev = wrist_t.left_hand.palm_position_device
                palm_l_cam_t = np.array(T_rgb_camera_device @ palm_l_dev).flatten()
                ee_pose_obs_t[:3] = rotation_matrix @ palm_l_cam_t
            if wrist_t.right_hand is not None:
                palm_r_dev = wrist_t.right_hand.palm_position_device
                palm_r_cam_t = np.array(T_rgb_camera_device @ palm_r_dev).flatten()
                ee_pose_obs_t[3:] = rotation_matrix @ palm_r_cam_t

        actions_t = np.zeros((HORIZON, ac_dim), dtype=np.float32)

        # Horizon rollout: *never* drop whole sample, only leave zeros where missing
        for offset in range(HORIZON):
            idx = t + offset * STEP
            if idx >= frame_length:
                break

            ts_ns = stream_timestamps_ns["rgb"][idx]
            wrist_off = mps_data_provider.get_wrist_and_palm_pose(
                ts_ns, time_query_closest
            )
            pose_off = mps_data_provider.get_closed_loop_pose(
                ts_ns, time_query_closest
            )

            if wrist_off is None or pose_off is None:
                continue

            cam_mat_off = build_camera_matrix(vrs_data_provider, pose_off)

            if hand == "right":
                if wrist_off.right_hand is None:
                    continue
                palm_dev = wrist_off.right_hand.palm_position_device
                palm_cam = (transform @ palm_dev).T
                palm_cam_h = np.concatenate([palm_cam, np.ones((1, 1))], axis=1)
                world = (cam_mat_off @ palm_cam_h.T).T  # (1,4)
                palm_in_cam_t = (camera_t_inv @ world.T).T[0, :3]
                actions_t[offset, :] = palm_in_cam_t
            elif hand == "left":
                if wrist_off.left_hand is None:
                    continue
                palm_dev = wrist_off.left_hand.palm_position_device
                palm_cam = (transform @ palm_dev).T
                palm_cam_h = np.concatenate([palm_cam, np.ones((1, 1))], axis=1)
                world = (cam_mat_off @ palm_cam_h.T).T
                palm_in_cam_t = (camera_t_inv @ world.T).T[0, :3]
                actions_t[offset, :] = palm_in_cam_t
            else:  # bimanual
                have_l = wrist_off.left_hand is not None
                have_r = wrist_off.right_hand is not None
                if not (have_l or have_r):
                    continue

                if have_l:
                    palm_l_dev = wrist_off.left_hand.palm_position_device
                    palm_l_cam = (transform @ palm_l_dev).T
                    palm_l_cam_h = np.concatenate(
                        [palm_l_cam, np.ones((1, 1))], axis=1
                    )
                    world_l = (cam_mat_off @ palm_l_cam_h.T).T
                    cam_l_t = (camera_t_inv @ world_l.T).T[0, :3]
                else:
                    cam_l_t = np.zeros(3, dtype=np.float32)

                if have_r:
                    palm_r_dev = wrist_off.right_hand.palm_position_device
                    palm_r_cam = (transform @ palm_r_dev).T
                    palm_r_cam_h = np.concatenate(
                        [palm_r_cam, np.ones((1, 1))], axis=1
                    )
                    world_r = (cam_mat_off @ palm_r_cam_h.T).T
                    cam_r_t = (camera_t_inv @ world_r.T).T[0, :3]
                else:
                    cam_r_t = np.zeros(3, dtype=np.float32)

                actions_t[offset, :] = np.concatenate([cam_l_t, cam_r_t], axis=0)

        # Rotate actions to EgoMimic convention
        if ac_dim == 3:
            rotated_actions_t = (rotation_matrix @ actions_t.T).T
        else:
            rotated_actions_t = actions_t.copy()
            rotated_actions_t[:, :3] = (rotation_matrix @ actions_t[:, :3].T).T
            rotated_actions_t[:, 3:] = (rotation_matrix @ actions_t[:, 3:].T).T

        actions_list.append(rotated_actions_t)
        imgs_list.append(img_t)
        ee_pose_list.append(ee_pose_obs_t)

    if len(actions_list) == 0:
        return np.zeros((0, HORIZON, ac_dim)), \
               np.zeros((0, 1, 1, 3), dtype=np.uint8), \
               np.zeros((0, ac_dim))

    actions = np.stack(actions_list, axis=0)
    front_img_1 = np.stack(imgs_list, axis=0)
    ee_pose = np.stack(ee_pose_list, axis=0)

    # Filter by FOV / jumps (same as before, but no additional frame skipping)
    ac_dim = actions.shape[-1]
    actions_flat = actions.reshape((-1, 3))
    px = cam_frame_to_cam_pixels(
        transform_actions(actions_flat), ARIA_INTRINSICS
    )
    px = px.reshape((-1, HORIZON, ac_dim))

    # if ac_dim == 3:
    #     bad_data_mask = (
    #         (px[:, :, 0] < -50)
    #         | (px[:, :, 0] > 690)
    #         | (px[:, :, 1] < -50)
    #         | (px[:, :, 1] > 530)
    #     )
    # else:  # 6
    #     BUFFER = 0
    #     bad_data_mask = (
    #         (px[:, :, 0] < 0 - BUFFER)
    #         | (px[:, :, 0] > 640 + BUFFER)
    #         | (px[:, :, 1] < 0)
    #         | (px[:, :, 3] < 0 - BUFFER)
    #         | (px[:, :, 3] > 640 + BUFFER)
    #         | (px[:, :, 4] < 0)
    #     )
    #     px_diff = np.diff(px, axis=1)
    #     px_diff = np.concatenate(
    #         (px_diff, np.zeros((px_diff.shape[0], 1, px_diff.shape[-1]))),
    #         axis=1,
    #     )
    #     px_diff = np.abs(px_diff)
    #     bad_data_mask = bad_data_mask | np.any(px_diff > 100, axis=2)

    # bad_data_mask = np.any(bad_data_mask, axis=1)

    # actions = actions[~bad_data_mask]
    # front_img_1 = front_img_1[~bad_data_mask]
    # ee_pose = ee_pose[~bad_data_mask]

    return actions, front_img_1, ee_pose

def transform_ee_pose(ee_pose):
    if ee_pose.shape[1] == 3:
        ee_pose[:, 0] *= -1  # Multiply x by -1
        ee_pose[:, 1] *= -1  # Multiply y by -1
    elif ee_pose.shape[1] == 6:
        ee_pose[:, 0] *= -1  # Multiply x by -1 for first set
        ee_pose[:, 1] *= -1  # Multiply y by -1 for first set
        ee_pose[:, 3] *= -1  # Multiply x by -1 for second set
        ee_pose[:, 4] *= -1  # Multiply y by -1 for second set

    return ee_pose


def transform_actions(actions):
    print("Transforming coordinates for actions and ee_pose")

    if actions.shape[1] == 3:
        actions[:, 0] *= -1  # Multiply x by -1
        actions[:, 1] *= -1  # Multiply y by -1
    elif actions.shape[1] == 6:
        actions[:, 0] *= -1  # Multiply x by -1 for first set
        actions[:, 1] *= -1  # Multiply y by -1 for first set
        actions[:, 3] *= -1  # Multiply x by -1 for second set
        actions[:, 4] *= -1  # Multiply y by -1 for second set
    elif actions.shape[1] == 30:
        for i in range(10):
            actions[:, 3 * i] *= -1  # Multiply x by -1 for each set
            actions[:, 3 * i + 1] *= -1  # Multiply y by -1 for each set
    elif actions.shape[1] == 60:
        for i in range(20):
            actions[:, 3 * i] *= -1  # Multiply x by -1 for each set
            actions[:, 3 * i + 1] *= -1  # Multiply y by -1 for each set

    return actions


def get_bounds(binary_image):
    """
    Get the bounding box of the hand mask
    binary_image: np.array of shape (h, w)

    Returns: min_x, max_x, min_y, max_y
    """
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary image
    # _,binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store max and min x and y values
    max_x = max_y = 0
    min_x = min_y = float('inf')

    if len(contours) == 0:
        return None, None, None, None

    # Loop through all contours to find max and min x and y values
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    return min_x, max_x, min_y, max_y

def line_on_hand(images, masks, arm):
    """
    Draw a line on the hand
    images: np.array of shape (n, h, w, c)
    masks: np.array of shape (n, h, w)
    arm: str, "left" or "right"
    """
    overlayed_imgs = np.zeros_like(images)
    for k, (image, mask) in enumerate(zip(images, masks)):
        min_x, max_x, min_y, max_y = get_bounds(mask.astype(np.uint8))
        if min_x is None:
            overlayed_imgs[k] = image
            continue

        gamma = 0.8
        alpha = 0.2
        scale = max_y - min_y
        min_x = int(max_x + gamma * (min_x - max_x))
        min_y = int(max_y + gamma * (min_y - max_y))
        max_x = int(max_x - scale * alpha)

        if arm == "right":
            line_image = cv2.line(image.copy(), (min_x,min_y),(max_x,max_y),color=(255,0,0), thickness=25)
        elif arm == "left":
            line_image = cv2.line(image.copy(), (min_x,max_y),(max_x,min_y),color=(255,0,0), thickness=25)
        else:
            raise ValueError(f"Invalid arm: {arm}")
        overlayed_imgs[k] = line_image
    
    return overlayed_imgs

def project_ee_to_pixels(ee_pose):
    """
    Project ee_pose (N,3) or (N,6) assumed to be in camera frame
    into pixel coordinates using ARIA_INTRINSICS.

    Returns:
      (N,2) if ee_pose is (N,3): [u, v]
      (N,4) if ee_pose is (N,6): [u_l, v_l, u_r, v_r]
    """
    if ee_pose.size == 0:
        return ee_pose

    if ee_pose.shape[1] == 3:
        # (N,3) -> (N,2)
        px = cam_frame_to_cam_pixels(ee_pose, ARIA_INTRINSICS)  # expects (N,3)
        return px.astype(np.int32)

    if ee_pose.shape[1] == 6:
        left = ee_pose[:, :3]
        right = ee_pose[:, 3:6]
        px_l = cam_frame_to_cam_pixels(left, ARIA_INTRINSICS)   # (N,2)
        px_r = cam_frame_to_cam_pixels(right, ARIA_INTRINSICS)  # (N,2)
        px = np.concatenate([px_l, px_r], axis=1)               # (N,4)
        return px.astype(np.int32)

    # Fallback
    return np.zeros((ee_pose.shape[0], 2), dtype=np.int32)

def draw_ee_on_images(images, ee_pose_px):
    """
    images: (N, H, W, 3), uint8
    ee_pose_px: (N,2) or (N,4), float
    Returns: (N, H, W, 3) with points drawn.
    """
    out = images.copy()
    N = images.shape[0]
    H, W = images.shape[1], images.shape[2]

    for i in range(N):
        img = out[i]
        pts = ee_pose_px[i]

        if pts.shape[0] == 2 or pts.shape[0] == 3:  # [u, v]
            u, v = int(round(pts[0])), int(round(pts[1]))
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (u, v), 6, (0, 255, 0), -1)

        elif pts.shape[0] == 4:  # [u_l, v_l, u_r, v_r]
            u_l, v_l, u_r, v_r = map(lambda x: int(round(x)), pts)
            if 0 <= u_l < W and 0 <= v_l < H:
                cv2.circle(img, (u_l, v_l), 6, (0, 255, 0), -1)
            if 0 <= u_r < W and 0 <= v_r < H:
                cv2.circle(img, (u_r, v_r), 6, (0, 0, 255), -1)

        elif pts.shape[0] == 6:
            u_l, v_l, _, u_r, v_r, _ = map(lambda x: int(round(x)), pts)
            if 0 <= u_l < W and 0 <= v_l < H:
                cv2.circle(img, (u_l, v_l), 6, (0, 255, 0), -1)
            if 0 <= u_r < W and 0 <= v_r < H:
                cv2.circle(img, (u_r, v_r), 6, (0, 0, 255), -1)

        out[i] = img

    return out

def sam_processing(dataset, debug=False):
    """
    Applying masking to all images in the dataset

    dataset: path to the hdf5 file
    """
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    with h5py.File(dataset, "r+") as data:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(len(data["data"].keys()))):
                demo = data[f"data/demo_{i}"]
                imgs = demo["obs/front_img_1"][:]
                ee_poses = demo["obs/ee_pose"][:]
                H, W = imgs.shape[1], imgs.shape[2]

                # 1. Detect invalid ee_poses: NaN or all zeros
                invalid = np.isnan(ee_poses).any(axis=1) | (np.linalg.norm(ee_poses, axis=1) == 0)
                
                # 2. Project to pixels to check if points fall outside image bounds
                try:
                    ee_poses_px = cam_frame_to_cam_pixels(ee_poses, ARIA_INTRINSICS)
                    out_of_bounds = (ee_poses_px[:, 0] < 0) | (ee_poses_px[:, 0] >= W) | \
                                    (ee_poses_px[:, 1] < 0) | (ee_poses_px[:, 1] >= H)
                    # Test if ee_poses_px is not 0,0
                    corner_case = (ee_poses_px[:, 0] == 0) & (ee_poses_px[:, 1] == 0)

                    invalid = invalid | out_of_bounds | corner_case
                except Exception:
                    pass # Fallback in case projection fails

                if invalid.all():
                    # all invalid -> null masks
                    raw_masks = np.zeros((imgs.shape[0], H, W), dtype=bool)
                    masked_imgs = imgs
                    overlayed_imgs = imgs
                else:
                    try:
                        overlayed_imgs, masked_imgs, raw_masks = sam.get_hand_mask_line_batched(
                            imgs, ee_poses, ARIA_INTRINSICS, debug=debug
                        )
                        
                        # 3. Wipe out generated masks for strictly invalid frames
                        raw_masks[invalid] = False
                        masked_imgs[invalid] = imgs[invalid]
                        overlayed_imgs[invalid] = imgs[invalid]
                        
                    except Exception:
                        raw_masks = np.zeros((imgs.shape[0], H, W), dtype=bool)
                        masked_imgs = imgs
                        overlayed_imgs = imgs

                if "front_img_1_masked" in demo["obs"]:
                    del demo["obs/front_img_1_masked"]
                if "front_img_1_line" in demo["obs"]:
                    del demo["obs/front_img_1_line"]
                if "front_img_1_mask" in demo["obs"]:
                    del demo["obs/front_img_1_mask"]

                demo["obs"].create_dataset(
                    "front_img_1_masked",
                    data=masked_imgs,
                    chunks=(1, 480, 640, 3),
                )
                demo["obs"].create_dataset(
                    "front_img_1_mask",
                    data=raw_masks,
                    chunks=(1, 480, 640),
                    dtype=bool,
                )
                demo["obs"].create_dataset(
                    "front_img_1_line",
                    data=overlayed_imgs,
                    chunks=(1, 480, 640, 3),
                )


def main(args):
    filenames = [f for f in os.listdir(args.dataset) if f.endswith(".vrs")]
    mps_paths = [
        os.path.join(args.dataset, "mps_" + filename.split(".")[0] + "_vrs")
        for filename in filenames
    ]

    if args.debug:
        filenames = filenames[0:2]
        mps_paths = mps_paths[0:2]
    
    with h5py.File(args.out, "w") as f:
        if args.hand == "left" or args.hand == "right":
            ac_dim = 3
        elif args.hand == "bimanual":
            ac_dim = 6

        demo_index = 0
        data = f.create_group(f"data")
        data.attrs["env_args"] = json.dumps({})
        print(f"Using {args.hand} data")
        for j, filename in enumerate(filenames):
            print(f"Adding {filename} to hdf5 file")
            actions, front_img_1, ee_pose = single_file_conversion(
                args.dataset, mps_paths[j], filename, args.hand
            )
            # actions, ee_pose = transform_actions(actions), transform_ee_pose(ee_pose)
            actions, ee_pose = actions, ee_pose
            ee_pose_px = project_ee_to_pixels(ee_pose)
            N = actions.shape[0]
            print(f"{N} frames in vrs file")
            chunk_size = 50  # Define chunk size
            for i in range(0, N, chunk_size):
                # print(i)
                group = data.create_group(f"demo_{demo_index}")
                # group.create_dataset("label", data=np.array([1]))
                # if args.prestack:
                ac_reshape = actions[i : i + chunk_size].reshape(-1, HORIZON, ac_dim)
                group.create_dataset("actions_xyz", data=ac_reshape)

                ac_reshape_interp = interpolate_arr(ac_reshape, 100)
                group.create_dataset("actions_xyz_act", data=ac_reshape_interp)
                    
                # else:
                #     group.create_dataset("actions", data=actions[i : i + chunk_size])
                group.attrs["num_samples"] = group["actions_xyz"].shape[0]
                group.create_dataset(
                    "obs/front_img_1", data=front_img_1[i : i + chunk_size]
                )
                group.create_dataset("obs/ee_pose", data=ee_pose[i : i + chunk_size])

                if args.debug:
                    group.create_dataset(
                        "obs/ee_pose_px", data=ee_pose_px[i : i + chunk_size]
                    )
                    debug_imgs = draw_ee_on_images(front_img_1[i : i + chunk_size], ee_pose_px[i : i + chunk_size])
                    group.create_dataset("obs/front_img_1_ee_debug", data=debug_imgs)
                demo_index += 1
            print(f"Completed adding {filename}")
            # break

    split_train_val_from_hdf5(hdf5_path=args.out, val_ratio=0.2)
    

    ## Apply masking
    if args.mask:
        print("Starting Masking")
        sam_processing(args.out, args.debug)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to folder containing vrs and mps",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="output file path",
    )
    parser.add_argument(
        "--hand",
        type=str,
        help="left; right; bimanual",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if true, debug runs for two files only. Defaults to False",
    )
    parser.add_argument(
        "--mask", action="store_true"
    )
    # parser.add_argument(
    #     "--prestack", action="store_true", help="if true, stacks actions in Tx3"
    # )

    args = parser.parse_args()

    assert args.hand in ["left", "right", "bimanual"], "Must provide the correct key (left, right, bimanual)"
    assert args.dataset is not None, "Must provide correct dataset folder"
    assert args.out is not None, "Must provide output file path"

    dataset_path = args.dataset

    main(args)
    