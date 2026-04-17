import os

folder_path = os.path.join(os.path.dirname(__file__))

import numpy as np
import cv2
import argparse
import json
import h5py
from tqdm import tqdm

from egomimic.utils.egomimicUtils import (
    # WIDE_LENS_ROBOT_LEFT_K,
    # WIDE_LENS_ROBOT_LEFT_D,
    ARIA_INTRINSICS
)

from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from pupil_apriltags import Detector


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

    # TODO get intrinsics
    # with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r") as f:
    #     intrinsics = json.load(f)
    # TODO: THESE ARE JUST TEMP VALUES
    intrinsics = ARIA_INTRINSICS
    intrinsics = {
        "color": {
            "fx": intrinsics[0, 0],
            "fy": intrinsics[1, 1],
            "cx": intrinsics[0, 2],
            "cy": intrinsics[1, 2],
        }
    }

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
            # img = cv2.undistort(
            #     img, WIDE_LENS_ROBOT_LEFT_K[:, :3], WIDE_LENS_ROBOT_LEFT_D
            # )

            # pupil-apriltags requires grayscale images
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Extract [fx, fy, cx, cy] from the intrinsics matrix
            K = intrinsics["color"]
            camera_params = [K['fx'], K['fy'], K['cx'], K['cy']]

            # Assuming you instantiated: april_detector = pupil_apriltags.Detector()
            detect_result = april_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=0.1393,
            )

            if len(detect_result) != 1:
                print(f"wrong detection, skipping img {t}")
                if args.debug:
                    plt.imsave(f"calibration_imgs_3/{t}_fail.png", img)

                continue



            count += 1
            print(list(demo["obs"].keys()))
            pose = demo["obs/ee_pose_robot_frame"][t]
            print("Pose:", pose)
            if pose.shape != (7,):
                raise ValueError(f"Expected pose shape (7,), but got {pose.shape}. Please check the pose format.")
            pos = pose[0:3]
            rot = Rot.from_quat(pose[3:])

            R_base2gripper_list.append(rot.as_matrix().T)
            t_base2gripper_list.append(
                -rot.as_matrix().T @ np.array(pos)[:, np.newaxis]
            )

            R_target2cam_list.append(detect_result[0].pose_R)
            pose_t = detect_result[0].pose_t


            bounding_box_corners = detect_result[0].corners
            # draw bounding box on img and save
            if args.debug:
                # pupil-apriltags doesn't have vis_tag, draw corners manually if needed
                for j in range(4):
                    pt1 = (int(bounding_box_corners[j][0]), int(bounding_box_corners[j][1]))
                    pt2 = (int(bounding_box_corners[(j + 1) % 4][0]), int(bounding_box_corners[(j + 1) % 4][1]))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    # reproject the tag center and draw it
                    tag_px = ARIA_INTRINSICS @ np.append(pose_t, 1.0)
                    tag_px = tag_px / tag_px[2]
                    tag_px = (int(tag_px[0]), int(tag_px[1]))
                    cv2.circle(img, tag_px, 5, (255, 0, 0), -1)
                plt.imsave(f"calibration_imgs_3/{t}_detection.png", img)

            # if args.debug:
            #     print("Detected: ", pose_t, T.quat2axisangle(T.mat2quat(detect_result[0].pose_R)))

            t_target2cam_list.append(pose_t)

    print(f"==========Using {count} images================")

    for method in [
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_DANIILIDIS,
        cv2.CALIB_HAND_EYE_ANDREFF,
        cv2.CALIB_HAND_EYE_HORAUD
    ]:
        R, t = cv2.calibrateHandEye(
            R_base2gripper_list,
            t_base2gripper_list,
            R_target2cam_list,
            t_target2cam_list,
            method=method,
        )
        # print("Rotation matrix: ", R.round(3))
        # print("Axis Angle: ", T.quat2axisangle(T.mat2quat(R)))
        # print("Quaternion: ", T.mat2quat(R))
        # print("Translation: ", t.T.round(3))
        fullT = np.concatenate((R, t), axis=1)
        fullT = np.concatenate((fullT, np.array([[0, 0, 0, 1]])), axis=0)
        print("T: ", repr(fullT))

    print("==============================")

    if args.store_matrix:
        store_matrix(args.h5py_path, R, t.T)


if __name__ == "__main__":
    main()
