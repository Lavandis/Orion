import numpy as np
import pandas as pd
import torch


def resolve_seq_len(seq_len_seconds: float, fps: float) -> int:
    """Convert a fixed horizon in seconds into discrete rollout steps."""
    if seq_len_seconds <= 0:
        raise ValueError("seq_len_seconds must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    return max(1, int(round(seq_len_seconds * fps)))


def get_time_axis(df: pd.DataFrame, fallback_dt: float) -> np.ndarray:
    """Return a time axis from the CSV or synthesize one from dt."""
    for candidate in ["Time_Sec", "Time"]:
        if candidate in df.columns:
            return df[candidate].to_numpy(dtype=np.float32)
    return np.arange(len(df), dtype=np.float32) * np.float32(fallback_dt)


def infer_dt_from_df(df: pd.DataFrame, fallback_dt: float) -> float:
    """Infer the discrete time step from the dataframe time axis."""
    time_axis = get_time_axis(df, fallback_dt)
    if len(time_axis) < 2:
        return fallback_dt

    dt = float(np.median(np.diff(time_axis)))
    return dt if dt > 0 else fallback_dt


def resolve_supervision_indices(
    seq_len: int,
    source_fps: float,
    supervision_fps: float | None,
) -> torch.Tensor:
    """Build frame indices used for sparse supervision while keeping full-resolution rollout."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")

    if supervision_fps is None or supervision_fps >= source_fps:
        return torch.arange(seq_len, dtype=torch.long)
    if supervision_fps <= 0:
        raise ValueError("supervision_fps must be positive.")

    stride = max(1, int(round(source_fps / supervision_fps)))
    indices = torch.arange(stride - 1, seq_len, stride, dtype=torch.long)
    if indices.numel() == 0 or indices[-1].item() != seq_len - 1:
        indices = torch.cat([indices, torch.tensor([seq_len - 1], dtype=torch.long)])
    return torch.unique(indices, sorted=True)


def apply_sparse_supervision(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    supervision_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the supervised frames from predicted and target trajectories."""
    if supervision_indices.device != pred_traj.device:
        supervision_indices = supervision_indices.to(pred_traj.device)
    return (
        pred_traj.index_select(1, supervision_indices),
        target_traj.index_select(1, supervision_indices),
    )


def get_theta_omega(
    df: pd.DataFrame, fallback_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load theta / omega, preferring stored omega when available."""
    angle_column = None
    for candidate in ["theta_rad", "Angle_rad"]:
        if candidate in df.columns:
            angle_column = candidate
            break

    if angle_column is not None:
        theta = df[angle_column].to_numpy(dtype=np.float32)
    else:
        theta = df.iloc[:, -1].to_numpy(dtype=np.float32)

    if "omega_rad_s" in df.columns:
        omega = df["omega_rad_s"].to_numpy(dtype=np.float32)
    else:
        time_axis = get_time_axis(df, fallback_dt)
        omega = np.gradient(theta, time_axis).astype(np.float32)

    return theta, omega
