import numpy as np
import pandas as pd


def resolve_seq_len(seq_len_seconds: float, fps: float) -> int:
    """Convert a fixed horizon in seconds into discrete rollout steps."""
    if seq_len_seconds <= 0:
        raise ValueError("seq_len_seconds must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    return max(1, int(round(seq_len_seconds * fps)))


def get_time_axis(df: pd.DataFrame, fallback_dt: float) -> np.ndarray:
    """Return a time axis from the CSV or synthesize one from dt."""
    if "Time_Sec" in df.columns:
        return df["Time_Sec"].to_numpy(dtype=np.float32)
    return np.arange(len(df), dtype=np.float32) * np.float32(fallback_dt)


def get_theta_omega(
    df: pd.DataFrame, fallback_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load theta / omega, preferring stored omega when available."""
    if "theta_rad" in df.columns:
        theta = df["theta_rad"].to_numpy(dtype=np.float32)
    else:
        theta = df.iloc[:, -1].to_numpy(dtype=np.float32)

    if "omega_rad_s" in df.columns:
        omega = df["omega_rad_s"].to_numpy(dtype=np.float32)
    else:
        time_axis = get_time_axis(df, fallback_dt)
        omega = np.gradient(theta, time_axis).astype(np.float32)

    return theta, omega
