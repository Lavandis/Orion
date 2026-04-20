import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import ODEDataset
from src.utils import apply_sparse_supervision, resolve_supervision_indices
from src.utils.metrics import calculate_mse


def train_panorama(
    model: nn.Module,
    train_df: pd.DataFrame,
    train_stages: list[dict],
    config: dict,
    device: torch.device,
    dt: float,
):
    """Train PANORAMA with trajectory and velocity supervision."""

    train_cfg = config["train"]
    lr = train_cfg["lr"]
    weight_decay = train_cfg["weight_decay"]
    batch_size = train_cfg["batch_size"]
    theta_loss_weight = train_cfg.get("theta_loss_weight", 1.0)
    omega_loss_weight = train_cfg.get("omega_loss_weight", 0.5)
    fa_reg_weight = train_cfg.get("fa_reg_weight", 0.0)

    total_epochs = sum(stage["epochs"] for stage in train_stages)
    warmup_epochs = config["multiplier_method"]["warmup_epochs"]
    source_fps = config["system"]["fps"]
    supervision_fps = train_cfg.get("supervision_downsample_to_fps")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"Start training for {total_epochs} epochs | "
        f"batch_size={batch_size} | stages={len(train_stages)}"
    )
    if supervision_fps is None:
        print(f"Sparse supervision: disabled | source_fps={source_fps}")
    else:
        print(
            f"Sparse supervision: source_fps={source_fps} -> supervision_fps={supervision_fps}"
        )

    global_epoch = 0
    model.train()
    for stage_idx, stage in enumerate(train_stages, start=1):
        stage_seq_len = stage["seq_len"]
        stage_seq_len_seconds = stage["seq_len_seconds"]
        stage_epochs = stage["epochs"]
        stage_dataset = ODEDataset(train_df, seq_len=stage_seq_len, dt=dt)
        stage_loader = DataLoader(
            stage_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
        )
        supervision_indices = resolve_supervision_indices(
            stage_seq_len,
            source_fps=source_fps,
            supervision_fps=supervision_fps,
        )

        print(
            f"Stage {stage_idx}/{len(train_stages)} | "
            f"{stage_seq_len_seconds:.2f}s -> {stage_seq_len} steps | "
            f"epochs={stage_epochs} | supervised_frames={len(supervision_indices)}"
        )

        for stage_epoch in range(stage_epochs):
            global_epoch += 1
            total_loss = 0.0
            total_theta_loss = 0.0
            total_omega_loss = 0.0
            total_fa_penalty = 0.0

            fa_weight = 0.0 if (global_epoch - 1) < warmup_epochs else 1.0

            for init_state, target_traj in stage_loader:
                init_state = init_state.to(device)
                target_traj = target_traj.to(device)

                optimizer.zero_grad()

                pred_traj, fa_norm = model(init_state, stage_seq_len)
                pred_traj, target_traj = apply_sparse_supervision(
                    pred_traj,
                    target_traj,
                    supervision_indices,
                )

                pred_theta = pred_traj[:, :, 0]
                target_theta = target_traj[:, :, 0]
                pred_omega = pred_traj[:, :, 1]
                target_omega = target_traj[:, :, 1]

                theta_loss = calculate_mse(pred_theta, target_theta)
                omega_loss = calculate_mse(pred_omega, target_omega)
                data_loss = (
                    theta_loss_weight * theta_loss + omega_loss_weight * omega_loss
                )
                loss = data_loss + fa_weight * fa_reg_weight * fa_norm

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_theta_loss += theta_loss.item()
                total_omega_loss += omega_loss.item()
                total_fa_penalty += fa_norm.item()

            num_batches = len(stage_loader)
            avg_loss = total_loss / num_batches
            avg_theta_loss = total_theta_loss / num_batches
            avg_omega_loss = total_omega_loss / num_batches
            avg_fa_penalty = total_fa_penalty / num_batches

            is_log_epoch = (
                global_epoch == 1
                or global_epoch % 5 == 0
                or stage_epoch == stage_epochs - 1
            )
            if is_log_epoch:
                phase = "warmup" if (global_epoch - 1) < warmup_epochs else "joint"
                effective_fa_reg = fa_weight * fa_reg_weight
                print(
                    f"Epoch {global_epoch:02d}/{total_epochs} [{phase}] | "
                    f"stage={stage_idx} | "
                    f"loss={avg_loss:.6e} | "
                    f"theta_mse={avg_theta_loss:.6e} | "
                    f"omega_mse={avg_omega_loss:.6e} | "
                    f"fa_penalty={avg_fa_penalty:.6e} | "
                    f"fa_reg={effective_fa_reg:.2e}"
                )

    save_path = config["paths"]["model_save"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    return model
