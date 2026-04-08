import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.metrics import calculate_mse


def train_panorama(
    model: nn.Module,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Train PANORAMA with trajectory and velocity supervision."""

    train_cfg = config["train"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["lr"]
    weight_decay = train_cfg["weight_decay"]
    seq_len = train_cfg["seq_len"]
    theta_loss_weight = train_cfg.get("theta_loss_weight", 1.0)
    omega_loss_weight = train_cfg.get("omega_loss_weight", 0.5)
    fa_reg_weight = train_cfg.get("fa_reg_weight", 0.0)

    warmup_epochs = config["multiplier_method"]["warmup_epochs"]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"Start training for {epochs} epochs | "
        f"batch_size={train_cfg['batch_size']} | seq_len={seq_len}"
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_theta_loss = 0.0
        total_omega_loss = 0.0
        total_fa_penalty = 0.0

        fa_weight = 0.0 if epoch < warmup_epochs else 1.0

        for init_state, target_traj in train_loader:
            init_state = init_state.to(device)
            target_traj = target_traj.to(device)

            optimizer.zero_grad()

            pred_traj, fa_norm = model(init_state, seq_len)

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

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_theta_loss = total_theta_loss / num_batches
        avg_omega_loss = total_omega_loss / num_batches
        avg_fa_penalty = total_fa_penalty / num_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            phase = "warmup" if epoch < warmup_epochs else "joint"
            effective_fa_reg = fa_weight * fa_reg_weight
            print(
                f"Epoch {epoch + 1:02d}/{epochs} [{phase}] | "
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
