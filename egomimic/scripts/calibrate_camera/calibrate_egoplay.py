import os

folder_path = os.path.join(os.path.dirname(__file__))

import numpy as np
import cv2
import argparse
import json
import h5py
from tqdm import tqdm

from egomimic.utils.egomimicUtils import ARIA_INTRINSICS

from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from pupil_apriltags import Detector

import pytorch_kinematics as pk
import torch
import egomimic as _egomimic_mod

# ── FK setup ────────────────────────────────────────────────────────────────
_URDF_PATH = os.path.join(os.path.dirname(_egomimic_mod.__file__), "resources/model.urdf")
_FK_CHAIN = pk.build_serial_chain_from_urdf(open(_URDF_PATH).read(), "vx300s/ee_gripper_link")

# ViperX joint index mapping: drop shadow joints (indices 2, 4) → 7 DOF
_VIPERX_9_TO_7 = [0, 1, 3, 5, 6, 7, 8]

# ViperX joint limits in radians (7 DOF: waist, shoulder, elbow, forearm_roll,
# wrist_angle, wrist_rotate, gripper).  Used to convert normalized_100 → radians.
_VIPERX_LIMITS_RAD = np.array([
    [-3.14159, 3.14159],   # waist
    [-1.9199,  1.9199],    # shoulder
    [-1.9199,  1.9199],    # elbow
    [-3.14159, 3.14159],   # forearm_roll
    [-1.7453,  1.7453],    # wrist_angle
    [-3.14159, 3.14159],   # wrist_rotate
    [-0.0349,  0.0349],    # gripper
])

_APRILTAG_SIZE_M = 0.1393


def _joints_to_radians(q7: np.ndarray) -> np.ndarray:
    """Convert a (7,) joint vector to radians, auto-detecting units."""
    abs_max = np.abs(q7).max()
    if abs_max > 10.0:           # normalized_100 (LeRobot raw recording)
        lo, hi = _VIPERX_LIMITS_RAD[:, 0], _VIPERX_LIMITS_RAD[:, 1]
        return (q7 / 100.0 * (hi - lo) / 2.0 + (lo + hi) / 2.0).astype(np.float64)
    elif abs_max > 3.5:          # degrees
        return np.deg2rad(q7).astype(np.float64)
    else:                        # already radians
        return q7.astype(np.float64)


def _fk_rot_pos(demo, t):
    """
    Return (Rot, pos) for frame t.

    Prefer FK from obs/joint_positions so calibration uses the same kinematic
    path as SAM masking. Fall back to obs/ee_pose_robot_frame only if joints
    are unavailable.
    """
    if "obs/joint_positions" in demo:
        q_raw = demo["obs/joint_positions"][t]

        # Remove shadow motors if present (9 DOF → 7 DOF)
        if q_raw.shape[0] == 9:
            q7 = q_raw[_VIPERX_9_TO_7]
        else:
            q7 = q_raw   # already 7 DOF

        q_rad = _joints_to_radians(np.asarray(q7, dtype=np.float64))

        # FK via pytorch_kinematics — takes (1, 6) radians, returns 4×4 T_ee_in_base
        q_tensor = torch.tensor(q_rad[:6], dtype=torch.float32).unsqueeze(0)
        T_ee = _FK_CHAIN.forward_kinematics(q_tensor, end_only=True).get_matrix().squeeze(0).numpy()
        return Rot.from_matrix(T_ee[:3, :3]), T_ee[:3, 3]

    if "obs/ee_pose_robot_frame" in demo:
        pose = demo["obs/ee_pose_robot_frame"][t]
        return Rot.from_quat(pose[3:]), pose[0:3]

    raise KeyError("Expected either obs/joint_positions or obs/ee_pose_robot_frame in demo")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h5py-path",
        type=str,
    )

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--store-matrix", action="store_true")

    return parser.parse_args()


def store_matrix(path, R, t):
    file = h5py.File(path, "r+")

    for demo_name in file.keys():
        demo = file[demo_name]
        calib_matrix_group = demo.create_group("calibration_matrix")
        calib_matrix_group.create_dataset("rotation", data=R)
        calib_matrix_group.create_dataset("translation", data=t)

    print("Appended calibration matrix: ")
    print(R.round(3))
    print(t.round(3))
    print("==============================")


def main():
    args = parse_args()

    calib = h5py.File(args.h5py_path, "r+")

    april_detector = Detector()

    # Use the shared pinhole intrinsics that correspond to the stored 640x480
    # undistorted Aria frames.
    intrinsics_matrix = ARIA_INTRINSICS.astype(np.float64)
    intrinsics = {
        "color": {
            "fx": float(intrinsics_matrix[0, 0]),
            "fy": float(intrinsics_matrix[1, 1]),
            "cx": float(intrinsics_matrix[0, 2]),
            "cy": float(intrinsics_matrix[1, 2]),
        }
    }
    K_3x4 = intrinsics_matrix

    print(intrinsics)

    if args.debug:
        import os
        os.makedirs("calibration_imgs_3", exist_ok=True)

    R_base2gripper_list = []
    t_base2gripper_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    calib = calib["data"]
    count = 0
    for key in calib.keys():
        demo = calib[key]
        T, H, W, _ = demo["obs/front_img_1"].shape
        for t in tqdm(range(T)):

            img = demo["obs/front_img_1"][t]

            # pupil-apriltags requires grayscale images
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Extract [fx, fy, cx, cy] from the intrinsics matrix
            K = intrinsics["color"]
            camera_params = [K['fx'], K['fy'], K['cx'], K['cy']]

            detect_result = april_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=_APRILTAG_SIZE_M,
            )

            if len(detect_result) != 1:
                print(f"wrong detection, skipping img {t}")
                if args.debug:
                    plt.imsave(f"calibration_imgs_3/{t}_fail.png", img)

                continue

            count += 1

            rot, pos = _fk_rot_pos(demo, t)

            R_base2gripper_list.append(rot.as_matrix().T)
            t_base2gripper_list.append(
                -rot.as_matrix().T @ np.array(pos)[:, np.newaxis]
            )

            R_target2cam_list.append(detect_result[0].pose_R)
            pose_t = detect_result[0].pose_t


            bounding_box_corners = detect_result[0].corners
            # draw bounding box on img and save
            if args.debug:
                R_tc = detect_result[0].pose_R   # (3,3) target-to-cam rotation
                t_tc = detect_result[0].pose_t   # (3,1) target-to-cam translation

                # Detected corners in pixel space (green)
                for j in range(4):
                    pt1 = (int(bounding_box_corners[j][0]), int(bounding_box_corners[j][1]))
                    pt2 = (int(bounding_box_corners[(j + 1) % 4][0]), int(bounding_box_corners[(j + 1) % 4][1]))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                # Reproject tag CORNERS using K — the definitive intrinsics test (blue)
                half = _APRILTAG_SIZE_M / 2.0
                tag_corners_3d = np.array([
                    [-half, -half, 0],
                    [ half, -half, 0],
                    [ half,  half, 0],
                    [-half,  half, 0],
                ], dtype=np.float64)
                for corner_3d in tag_corners_3d:
                    p_cam = R_tc @ corner_3d + t_tc.flatten()
                    px = K_3x4 @ np.append(p_cam, 1.0)
                    px = px / px[2]
                    cv2.circle(img, (int(px[0]), int(px[1])), 5, (255, 0, 0), -1)

                # Reproject tag CENTER (blue dot, larger)
                tag_px = K_3x4 @ np.append(pose_t, 1.0)
                tag_px = tag_px / tag_px[2]
                cv2.circle(img, (int(tag_px[0]), int(tag_px[1])), 8, (255, 0, 0), 2)

                # Annotation: saved debug images are RGB, so the reprojected points
                # appear red after drawing with OpenCV on the in-memory image.
                cv2.putText(img, "Green=detected  Red=reprojected (K correct if they overlap)",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                plt.imsave(f"calibration_imgs_3/{t}_detection.png", img)

            t_target2cam_list.append(pose_t)

    print(f"==========Using {count} images================")

    hand_eye_methods = [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
        ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
    ]
    for method_name, method in hand_eye_methods:
        R, t = cv2.calibrateHandEye(
            R_base2gripper_list,
            t_base2gripper_list,
            R_target2cam_list,
            t_target2cam_list,
            method=method,
        )
        fullT = np.concatenate((R, t), axis=1)
        fullT = np.concatenate((fullT, np.array([[0, 0, 0, 1]])), axis=0)
        print(f"{method_name}: ", repr(fullT))

    print("==============================")

    if args.store_matrix:
        store_matrix(args.h5py_path, R, t.T)


if __name__ == "__main__":
    main()
