import sys
import numpy as np
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId

def main(vrs_path: str):
    prov = data_provider.create_vrs_data_provider(vrs_path)
    device_calibration = prov.get_device_calibration()

    # --- RGB stream: 214-1 ---
    rgb_stream_id = StreamId("214-1")
    rgb_label = prov.get_label_from_stream_id(rgb_stream_id)
    print("RGB label:", rgb_label)

    cam_calib = device_calibration.get_camera_calib(rgb_label)

    # Option A: raw projection params (model specific)
    # print("projection_type:", cam_calib.projection_type())
    print("projection_params:", cam_calib.projection_params())

    # Option B: derive a linear pinhole K matching your aria_utils usage
    lin_calib = calibration.get_linear_camera_calibration(
        480, 640, 133.25430222 * 2, rgb_label, cam_calib.get_transform_device_camera()
    )
    fx, fy = lin_calib.get_focal_lengths()
    cx, cy = lin_calib.get_principal_point()

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])

    print("Derived linear focal lengths fx, fy:", fx, fy)
    print("Derived linear principal point cx, cy:", cx, cy)
    print("Linear K:\n", K)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_intrinsics.py /path/to/file.vrs")
        sys.exit(1)
    main(sys.argv[1])