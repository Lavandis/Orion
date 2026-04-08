import glob
import os

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter


def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_savgol_window(length: int, window: int) -> int | None:
    if length < 3:
        return None

    max_odd = length if length % 2 == 1 else length - 1
    if max_odd < 3:
        return None

    window = min(window, max_odd)
    if window % 2 == 0:
        window -= 1

    return window if window >= 3 else None


def process_data():
    config = load_config()

    raw_dir = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)

    pivot_x = config["camera"]["pivot_x"]
    pivot_y = config["camera"]["pivot_y"]
    use_filter = config["camera"]["use_filter"]
    smooth_window = config["camera"]["smooth_window"]
    smooth_poly = config["camera"]["smooth_poly"]

    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return

    print(f"Found {len(csv_files)} raw CSV files. Start preprocessing...")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"  -> Processing: {filename}")

        df = pd.read_csv(file_path)
        required_columns = {"Frame", "Time_Sec", "Center_X", "Center_Y"}
        if not required_columns.issubset(df.columns):
            print(f"     Skip {filename}: missing required columns {required_columns}")
            continue

        dx = df["Center_X"].to_numpy(dtype=np.float32) - np.float32(pivot_x)
        dy = df["Center_Y"].to_numpy(dtype=np.float32) - np.float32(pivot_y)
        theta_raw = np.arctan2(dx, dy).astype(np.float32)
        time_sec = df["Time_Sec"].to_numpy(dtype=np.float32)

        filter_window = resolve_savgol_window(len(theta_raw), smooth_window)
        if use_filter and filter_window is not None and smooth_poly < filter_window:
            theta_processed = savgol_filter(
                theta_raw,
                window_length=filter_window,
                polyorder=smooth_poly,
            ).astype(np.float32)
        else:
            theta_processed = theta_raw

        omega_processed = np.gradient(theta_processed, time_sec).astype(np.float32)

        processed_df = pd.DataFrame(
            {
                "Frame": df["Frame"],
                "Time_Sec": df["Time_Sec"],
                "theta_rad": theta_processed,
                "omega_rad_s": omega_processed,
            }
        )

        save_path = os.path.join(processed_dir, filename)
        processed_df.to_csv(save_path, index=False)

    print(f"Preprocessing complete. Saved processed data to {processed_dir}")


if __name__ == "__main__":
    process_data()
