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
    ARIA_LINEAR_FOCAL,
    build_camera_matrix,
    undistort_to_linear,
    split_train_val_from_hdf5,
    slam_to_rgb,
)

import argparse

import json
import csv
import bisect

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    WIDE_LENS_HAND_LEFT_K,
    ARIA_INTRINSICS,
    ARIA_INTRINSICS_ROTATED,
    interpolate_keys,
    interpolate_arr
)
from egomimic.scripts.masking.utils import *

HORIZON = 10
STEP = 3
EGO_ROTATION_MATRIX = np.array(
    [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)
EGO_ROTATION_MATRIX_INV = EGO_ROTATION_MATRIX.T


def intrinsics_for_image_shape(image_shape):
    h, w = image_shape[:2]
    return np.array(
        [
            [ARIA_LINEAR_FOCAL, 0.0, w / 2.0, 0.0],
            [0.0, ARIA_LINEAR_FOCAL, h / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _valid_hand(hand_obj):
    return hand_obj is not None and getattr(hand_obj, "confidence", 0) > 0


def _get_palm_position_device(hand_obj):
    if hasattr(hand_obj, "get_palm_position_device"):
        return hand_obj.get_palm_position_device()
    return hand_obj.palm_position_device


def _get_wrist_position_device(hand_obj):
    if hasattr(hand_obj, "get_wrist_position_device"):
        return hand_obj.get_wrist_position_device()
    return hand_obj.wrist_position_device


def _apply_rotation_to_groups(arr, rotation_matrix):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    if arr.shape[-1] % 3 != 0:
        raise ValueError(f"Expected last dim to be a multiple of 3, got shape {arr.shape}")
    reshaped = arr.reshape(*arr.shape[:-1], -1, 3)
    rotated = np.einsum("ij,...gj->...gi", rotation_matrix, reshaped)
    return rotated.reshape(arr.shape)


def _load_valid_hand_timestamps_us(hand_tracking_results_path):
    valid = {"left": [], "right": []}
    with open(hand_tracking_results_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us = int(row["tracking_timestamp_us"])
            try:
                left_conf = float(row.get("left_tracking_confidence", -1))
            except Exception:
                left_conf = -1
            try:
                right_conf = float(row.get("right_tracking_confidence", -1))
            except Exception:
                right_conf = -1
            if left_conf > 0:
                valid["left"].append(ts_us)
            if right_conf > 0:
                valid["right"].append(ts_us)
    return valid


def _find_nearest_valid_timestamp_ns(valid_timestamps_us, target_timestamp_ns, max_delta_ms):
    if not valid_timestamps_us:
        return None
    target_us = target_timestamp_ns / 1000.0
    idx = bisect.bisect_left(valid_timestamps_us, target_us)
    best = None
    best_delta = None
    for cand_idx in (idx - 1, idx):
        if 0 <= cand_idx < len(valid_timestamps_us):
            cand_us = valid_timestamps_us[cand_idx]
            delta_us = abs(cand_us - target_us)
            if best_delta is None or delta_us < best_delta:
                best_delta = delta_us
                best = cand_us * 1000
    if best is None:
        return None
    if best_delta is not None and best_delta <= max_delta_ms * 1000.0:
        return int(best)
    return None


"""
Example usage
python aria_to_robomimic.py --dataset /coc/flash7/datasets/egoplay/oboo_aria_apr16/oboo_aria_apr16/ --out /coc/flash7/datasets/egoplay/oboo_aria_apr16/converted/oboo_aria_apr16_rightMimicplay.hdf5 --hand right
"""

# Load the VRS file


def single_file_conversion(
    dataset,
    mps_sample_path,
    filename,
    hand,
    rotate90="cw",
    crop_frames=50,
    max_rgb_frames=None,
    hand_time_tolerance_ms=33.0,
):
    """
    dataset: path to the dataset
    mps_sample_path: path to the mps sample
    filename: name of the vrs file
    hand: left, right, bimanual

    Returns: actions [N, HORIZON, ac_dim], front_img_1 [N, H, W, 3], ee_pose [N, ac_dim]
    """
    vrsfile = os.path.join(dataset, filename)

    hand_tracking_results_path = os.path.join(
        mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
    )

    provider = data_provider.create_vrs_data_provider(vrsfile)

    # Load hand tracking
    _ = mps.hand_tracking.read_hand_tracking_results(hand_tracking_results_path)
    valid_hand_timestamps_us = _load_valid_hand_timestamps_us(hand_tracking_results_path)

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
    if max_rgb_frames is not None:
        frame_length = min(frame_length, max_rgb_frames)
    print(f"Total RGB frames: {frame_length}")

    actions_list = []
    imgs_list = []
    ee_pose_list = []
    wrist_pose_list = []

    ac_dim = 3 if hand != "bimanual" else 6
    stats = {
        "total_rgb_frames": frame_length,
        "crop_frames_each_side": crop_frames,
        "candidate_frames": 0,
        "skipped_missing_reference_hand": 0,
        "skipped_rgb_read_error": 0,
        "skipped_missing_pose_t": 0,
        "kept_frames": 0,
    }

    center_px_test = ARIA_INTRINSICS @ np.array([0, 0, 1, 1])
    print(f"Optical center should be at: {center_px_test[:2]}")

    for t in range(crop_frames, max(crop_frames, frame_length - crop_frames)):
        stats["candidate_frames"] += 1
        if (t % 1000) == 0:
            print(f"{t} frames ingested")

        sample_timestamp_ns_t = stream_timestamps_ns["rgb"][t]

        # Get a valid hand pose near the RGB timestamp instead of requiring the
        # nearest overall hand-tracking sample to also be valid.
        ref_left_ns = _find_nearest_valid_timestamp_ns(
            valid_hand_timestamps_us["left"], sample_timestamp_ns_t, hand_time_tolerance_ms
        )
        ref_right_ns = _find_nearest_valid_timestamp_ns(
            valid_hand_timestamps_us["right"], sample_timestamp_ns_t, hand_time_tolerance_ms
        )

        if hand == "left":
            if ref_left_ns is None:
                stats["skipped_missing_reference_hand"] += 1
                continue
            wrist_t = mps_data_provider.get_hand_tracking_result(ref_left_ns, time_query_closest)
        elif hand == "right":
            if ref_right_ns is None:
                stats["skipped_missing_reference_hand"] += 1
                continue
            wrist_t = mps_data_provider.get_hand_tracking_result(ref_right_ns, time_query_closest)
        else:
            if ref_left_ns is None and ref_right_ns is None:
                stats["skipped_missing_reference_hand"] += 1
                continue
            ref_ns = ref_left_ns if ref_left_ns is not None else ref_right_ns
            wrist_t = mps_data_provider.get_hand_tracking_result(ref_ns, time_query_closest)

        if wrist_t is None:
            stats["skipped_missing_reference_hand"] += 1
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
            stats["skipped_rgb_read_error"] += 1
            continue

        img_t = undistort_to_linear(
            provider,
            stream_ids,
            raw_image=frame_rgb.to_numpy_array(),
            rotate_90=rotate90,
        )

        # Closed-loop pose at t (camera pose); if missing we cannot define camera frame
        pose_t = mps_data_provider.get_closed_loop_pose(
            sample_timestamp_ns_t, time_query_closest
        )
        if pose_t is None:
            stats["skipped_missing_pose_t"] += 1
            continue

        camera_matrix_t = build_camera_matrix(vrs_data_provider, pose_t)
        camera_t_inv = np.linalg.inv(camera_matrix_t)

        # ee_pose at reference frame in camera_t
        if hand == "right":
            palm_dev = _get_palm_position_device(wrist_t.right_hand)
            wrist_dev = _get_wrist_position_device(wrist_t.right_hand)
            # Ensure the evaluated 3D point strictly has shape (3,)
            palm_cam_t = np.array(T_rgb_camera_device @ palm_dev).flatten()
            wrist_cam_t = np.array(T_rgb_camera_device @ wrist_dev).flatten()

            # Rotate to EgoMimic convention
            ee_pose_obs_t = EGO_ROTATION_MATRIX @ palm_cam_t
            wrist_pose_obs_t = EGO_ROTATION_MATRIX @ wrist_cam_t
        elif hand == "left":
            palm_dev = _get_palm_position_device(wrist_t.left_hand)
            wrist_dev = _get_wrist_position_device(wrist_t.left_hand)
            palm_cam_t = np.array(T_rgb_camera_device @ palm_dev).flatten()
            wrist_cam_t = np.array(T_rgb_camera_device @ wrist_dev).flatten()
            ee_pose_obs_t = EGO_ROTATION_MATRIX @ palm_cam_t
            wrist_pose_obs_t = EGO_ROTATION_MATRIX @ wrist_cam_t
        else:
            ee_pose_obs_t = np.zeros(6, dtype=np.float32)
            wrist_pose_obs_t = np.zeros(6, dtype=np.float32)
            if _valid_hand(wrist_t.left_hand):
                palm_l_dev = _get_palm_position_device(wrist_t.left_hand)
                wrist_l_dev = _get_wrist_position_device(wrist_t.left_hand)
                palm_l_cam_t = np.array(T_rgb_camera_device @ palm_l_dev).flatten()
                wrist_l_cam_t = np.array(T_rgb_camera_device @ wrist_l_dev).flatten()
                ee_pose_obs_t[:3] = EGO_ROTATION_MATRIX @ palm_l_cam_t
                wrist_pose_obs_t[:3] = EGO_ROTATION_MATRIX @ wrist_l_cam_t
            if _valid_hand(wrist_t.right_hand):
                palm_r_dev = _get_palm_position_device(wrist_t.right_hand)
                wrist_r_dev = _get_wrist_position_device(wrist_t.right_hand)
                palm_r_cam_t = np.array(T_rgb_camera_device @ palm_r_dev).flatten()
                wrist_r_cam_t = np.array(T_rgb_camera_device @ wrist_r_dev).flatten()
                ee_pose_obs_t[3:] = EGO_ROTATION_MATRIX @ palm_r_cam_t
                wrist_pose_obs_t[3:] = EGO_ROTATION_MATRIX @ wrist_r_cam_t

        actions_t = np.zeros((HORIZON, ac_dim), dtype=np.float32)

        # Horizon rollout: *never* drop whole sample, only leave zeros where missing
        for offset in range(HORIZON):
            idx = t + offset * STEP
            if idx >= frame_length:
                break

            ts_ns = stream_timestamps_ns["rgb"][idx]
            off_left_ns = _find_nearest_valid_timestamp_ns(
                valid_hand_timestamps_us["left"], ts_ns, hand_time_tolerance_ms
            )
            off_right_ns = _find_nearest_valid_timestamp_ns(
                valid_hand_timestamps_us["right"], ts_ns, hand_time_tolerance_ms
            )
            query_ns = None
            if hand == "left":
                query_ns = off_left_ns
            elif hand == "right":
                query_ns = off_right_ns
            else:
                query_ns = off_left_ns if off_left_ns is not None else off_right_ns
            wrist_off = (
                mps_data_provider.get_hand_tracking_result(query_ns, time_query_closest)
                if query_ns is not None
                else None
            )
            pose_off = mps_data_provider.get_closed_loop_pose(
                ts_ns, time_query_closest
            )

            if wrist_off is None or pose_off is None:
                continue

            cam_mat_off = build_camera_matrix(vrs_data_provider, pose_off)

            if hand == "right":
                if not _valid_hand(wrist_off.right_hand):
                    continue
                palm_dev = _get_palm_position_device(wrist_off.right_hand)
                palm_cam = (transform @ palm_dev).T
                palm_cam_h = np.concatenate([palm_cam, np.ones((1, 1))], axis=1)
                world = (cam_mat_off @ palm_cam_h.T).T  # (1,4)
                palm_in_cam_t = (camera_t_inv @ world.T).T[0, :3]
                actions_t[offset, :] = palm_in_cam_t
            elif hand == "left":
                if not _valid_hand(wrist_off.left_hand):
                    continue
                palm_dev = _get_palm_position_device(wrist_off.left_hand)
                palm_cam = (transform @ palm_dev).T
                palm_cam_h = np.concatenate([palm_cam, np.ones((1, 1))], axis=1)
                world = (cam_mat_off @ palm_cam_h.T).T
                palm_in_cam_t = (camera_t_inv @ world.T).T[0, :3]
                actions_t[offset, :] = palm_in_cam_t
            else:  # bimanual
                have_l = _valid_hand(wrist_off.left_hand)
                have_r = _valid_hand(wrist_off.right_hand)
                if not (have_l or have_r):
                    continue

                if have_l:
                    palm_l_dev = _get_palm_position_device(wrist_off.left_hand)
                    palm_l_cam = (transform @ palm_l_dev).T
                    palm_l_cam_h = np.concatenate(
                        [palm_l_cam, np.ones((1, 1))], axis=1
                    )
                    world_l = (cam_mat_off @ palm_l_cam_h.T).T
                    cam_l_t = (camera_t_inv @ world_l.T).T[0, :3]
                else:
                    cam_l_t = np.zeros(3, dtype=np.float32)

                if have_r:
                    palm_r_dev = _get_palm_position_device(wrist_off.right_hand)
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
        rotated_actions_t = _apply_rotation_to_groups(actions_t, EGO_ROTATION_MATRIX)

        actions_list.append(rotated_actions_t)
        imgs_list.append(img_t)
        ee_pose_list.append(ee_pose_obs_t)
        wrist_pose_list.append(wrist_pose_obs_t)
        stats["kept_frames"] += 1

    if len(actions_list) == 0:
        return np.zeros((0, HORIZON, ac_dim)), \
               np.zeros((0, 1, 1, 3), dtype=np.uint8), \
               np.zeros((0, ac_dim)), \
               np.zeros((0, ac_dim)), stats

    actions = np.stack(actions_list, axis=0)
    front_img_1 = np.stack(imgs_list, axis=0)
    ee_pose = np.stack(ee_pose_list, axis=0)
    wrist_pose = np.stack(wrist_pose_list, axis=0)

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

    return actions, front_img_1, ee_pose, wrist_pose, stats

def transform_ee_pose(ee_pose):
    return _apply_rotation_to_groups(ee_pose, EGO_ROTATION_MATRIX_INV)


def transform_actions(actions):
    print("Transforming coordinates for actions and ee_pose")
    return _apply_rotation_to_groups(actions, EGO_ROTATION_MATRIX_INV)


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

def project_ee_to_pixels(ee_pose, intrinsics):
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
        px = cam_frame_to_cam_pixels(ee_pose, intrinsics)  # expects (N,3)
        return px.astype(np.int32)

    if ee_pose.shape[1] == 6:
        left = ee_pose[:, :3]
        right = ee_pose[:, 3:6]
        px_l = cam_frame_to_cam_pixels(left, intrinsics)   # (N,2)
        px_r = cam_frame_to_cam_pixels(right, intrinsics)  # (N,2)
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

def sam_processing(dataset, arm="right", debug=False):
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
                if "wrist_pose" in demo["obs"]:
                    ee_poses = demo["obs/wrist_pose"][:]
                else:
                    ee_poses = demo["obs/ee_pose"][:]
                ee_poses_camera = transform_ee_pose(ee_poses.copy())
                H, W = imgs.shape[1], imgs.shape[2]
                intrinsics = intrinsics_for_image_shape(imgs.shape[1:3])

                # 1. Detect invalid ee_poses: NaN or all zeros
                invalid = np.isnan(ee_poses_camera).any(axis=1) | (
                    np.linalg.norm(ee_poses_camera, axis=1) == 0
                )
                
                # 2. Project to pixels to check if points fall outside image bounds
                try:
                    ee_poses_px = cam_frame_to_cam_pixels(ee_poses_camera, intrinsics)
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
                            imgs, ee_poses_camera, intrinsics, arm=arm, debug=debug
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
                    chunks=(1, H, W, 3),
                )
                demo["obs"].create_dataset(
                    "front_img_1_mask",
                    data=raw_masks,
                    chunks=(1, H, W),
                    dtype=bool,
                )
                demo["obs"].create_dataset(
                    "front_img_1_line",
                    data=overlayed_imgs,
                    chunks=(1, H, W, 3),
                )


def main(args):
    filenames = sorted([f for f in os.listdir(args.dataset) if f.endswith(".vrs")])
    if args.file_contains:
        filenames = [f for f in filenames if args.file_contains in f]
    if args.start_file > 0:
        filenames = filenames[args.start_file:]
    if args.max_files is not None:
        filenames = filenames[: args.max_files]
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
            actions, front_img_1, ee_pose, wrist_pose, stats = single_file_conversion(
                args.dataset,
                mps_paths[j],
                filename,
                args.hand,
                args.rotate90,
                args.crop_frames,
                args.max_rgb_frames,
                args.hand_time_tolerance_ms,
            )
            # actions, ee_pose = transform_actions(actions), transform_ee_pose(ee_pose)
            actions, ee_pose = actions, ee_pose
            N = actions.shape[0]
            intrinsics = (
                intrinsics_for_image_shape(front_img_1.shape[1:3])
                if N > 0
                else ARIA_INTRINSICS
            )
            debug_pose = wrist_pose if wrist_pose.shape[0] == N else ee_pose
            ee_pose_px = project_ee_to_pixels(transform_ee_pose(debug_pose.copy()), intrinsics)
            print(f"{N} frames in vrs file")
            chunk_size = args.demo_chunk_size
            n_demos = int(np.ceil(N / chunk_size)) if chunk_size > 0 else 0
            print(
                "Conversion stats:",
                {
                    **stats,
                    "output_frames": int(N),
                    "demo_chunk_size": int(chunk_size),
                    "output_demos": int(n_demos),
                },
            )
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
                group.create_dataset("obs/wrist_pose", data=wrist_pose[i : i + chunk_size])

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
        sam_processing(args.out, args.hand, args.debug)




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
        "--max-files",
        type=int,
        default=None,
        help="Optional number of .vrs files to process.",
    )
    parser.add_argument(
        "--start-file",
        type=int,
        default=0,
        help="Optional zero-based index of the first .vrs file to process after sorting.",
    )
    parser.add_argument(
        "--file-contains",
        type=str,
        default=None,
        help="Optional substring filter on the .vrs filename for debugging a specific recording.",
    )
    parser.add_argument(
        "--max-rgb-frames",
        type=int,
        default=None,
        help="Optional cap on the number of RGB frames read from each file for faster debugging.",
    )
    parser.add_argument(
        "--crop-frames",
        type=int,
        default=50,
        help="Number of RGB frames to discard at the start and end of each file.",
    )
    parser.add_argument(
        "--mask", action="store_true"
    )
    parser.add_argument(
        "--demo-chunk-size",
        type=int,
        default=200,
        help="Number of kept frames per HDF5 demo chunk. Smaller values create more demos.",
    )
    parser.add_argument(
        "--rotate90",
        type=str,
        choices=["cw", "ccw", "none"],
        default="cw",
        help="Apply a 90 degree rotation to the undistorted RGB image. Use 'ccw' for Gen2 if the image is sideways the wrong way.",
    )
    parser.add_argument(
        "--hand-time-tolerance-ms",
        type=float,
        default=33.0,
        help="Allow matching an RGB frame to the nearest valid hand-tracking sample within this time window.",
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
    
