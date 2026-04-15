import pandas as pd
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python convert_hand_poses.py <path_to_hand_tracking_results.csv>")
    sys.exit(1)

input_path = sys.argv[1]
output_dir = os.path.dirname(input_path)

hand = pd.read_csv(input_path, sep=",")

# ── Output 1: hand_tracking_results.csv ──────────────────────────────────────
hand.to_csv(os.path.join(output_dir, "hand_tracking_results.csv"), sep=",", index=False)

# ── Output 2: wrist_and_palm_poses.csv ───────────────────────────────────────
wrist_palm = pd.DataFrame()
wrist_palm["tracking_timestamp_us"]    = hand["tracking_timestamp_us"]
wrist_palm["left_tracking_confidence"] = hand["left_tracking_confidence"]

wrist_palm["tx_left_wrist_device"] = hand["tx_left_device_wrist"]
wrist_palm["ty_left_wrist_device"] = hand["ty_left_device_wrist"]
wrist_palm["tz_left_wrist_device"] = hand["tz_left_device_wrist"]

PALM_LANDMARKS = [0, 5, 9, 13, 17]
for side in ["left", "right"]:
    for axis in ["x", "y", "z"]:
        cols = [f"t{axis}_{side}_landmark_{i}_device" for i in PALM_LANDMARKS]
        wrist_palm[f"t{axis}_{side}_palm_device"] = hand[cols].mean(axis=1)

wrist_palm["right_tracking_confidence"] = hand["right_tracking_confidence"]
wrist_palm["tx_right_wrist_device"] = hand["tx_right_device_wrist"]
wrist_palm["ty_right_wrist_device"] = hand["ty_right_device_wrist"]
wrist_palm["tz_right_wrist_device"] = hand["tz_right_device_wrist"]

for col in [
    "nx_left_palm_device",  "ny_left_palm_device",  "nz_left_palm_device",
    "nx_left_wrist_device", "ny_left_wrist_device", "nz_left_wrist_device",
    "nx_right_palm_device", "ny_right_palm_device", "nz_right_palm_device",
    "nx_right_wrist_device","ny_right_wrist_device","nz_right_wrist_device",
]:
    wrist_palm[col] = hand[col]

wrist_palm.to_csv(os.path.join(output_dir, "wrist_and_palm_poses.csv"), sep=",", index=False)

print("Done.")
print(f"  {output_dir}/hand_tracking_results.csv → {hand.shape[0]} rows")
print(f"  {output_dir}/wrist_and_palm_poses.csv  → {wrist_palm.shape[0]} rows")
