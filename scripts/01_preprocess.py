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


def extract_time_theta(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    time_column = None
    for candidate in ["Time_Sec", "Time"]:
        if candidate in df.columns:
            time_column = candidate
            break
    if time_column is None:
        raise ValueError("Missing time column. Expected one of: Time_Sec, Time")

    angle_column = None
    for candidate in ["theta_rad", "Angle_rad"]:
        if candidate in df.columns:
            angle_column = candidate
            break

    time_sec = df[time_column].to_numpy(dtype=np.float32)
    if angle_column is not None:
        theta_raw = df[angle_column].to_numpy(dtype=np.float32)
        return time_sec, theta_raw

    if {"Center_X", "Center_Y"}.issubset(df.columns):
        camera_cfg = config.get("camera", {})
        pivot_x = camera_cfg.get("pivot_x")
        pivot_y = camera_cfg.get("pivot_y")
        if pivot_x is None or pivot_y is None:
            raise ValueError(
                "Raw data uses Center_X/Center_Y, but pivot_x/pivot_y are not configured."
            )

        dx = df["Center_X"].to_numpy(dtype=np.float32) - np.float32(pivot_x)
        dy = df["Center_Y"].to_numpy(dtype=np.float32) - np.float32(pivot_y)
        theta_raw = np.arctan2(dx, dy).astype(np.float32)
        return time_sec, theta_raw

    raise ValueError(
        "Missing angle information. Expected Angle_rad/theta_rad or Center_X/Center_Y."
    )


def process_data():
    config = load_config()

    raw_dir = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)

    preprocess_cfg = config.get("preprocess", {})
    use_filter = preprocess_cfg.get("use_filter", False)
    smooth_window = preprocess_cfg.get("smooth_window", 11)
    smooth_poly = preprocess_cfg.get("smooth_poly", 3)

    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return

    print(f"Found {len(csv_files)} raw CSV files. Start preprocessing...")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"  -> Processing: {filename}")

        df = pd.read_csv(file_path)
        try:
            time_sec, theta_raw = extract_time_theta(df, config)
        except ValueError as exc:
            print(f"     Skip {filename}: {exc}")
            continue

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
        if "Frame" in df.columns:
            frame = df["Frame"].to_numpy()
        else:
            frame = np.arange(len(theta_processed), dtype=np.int64)

        processed_df = pd.DataFrame(
            {
                "Frame": frame,
                "Time_Sec": time_sec,
                "theta_rad": theta_processed,
                "omega_rad_s": omega_processed,
            }
        )

        save_path = os.path.join(processed_dir, filename)
        processed_df.to_csv(save_path, index=False)

    print(f"Preprocessing complete. Saved processed data to {processed_dir}")


if __name__ == "__main__":
    process_data()
