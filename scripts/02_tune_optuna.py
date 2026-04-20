import os
import sys

import optuna
import pandas as pd
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ODEDataset
from src.models import PANORAMA
from src.utils import (
    apply_sparse_supervision,
    calculate_mse,
    calculate_rmse_numpy,
    get_theta_omega,
    resolve_seq_len,
    resolve_supervision_indices,
)


def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_train_val_split(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = config["data"]["active_dataset"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset: {data_path}")

    full_df = pd.read_csv(data_path)
    train_ratio = config["data"].get("train_ratio", 0.75)
    train_end = int(len(full_df) * train_ratio)
    trainval_df = full_df.iloc[:train_end]

    tune_cfg = config.get("tune", {})
    val_ratio = tune_cfg.get("validation_ratio", 0.2)
    inner_train_end = int(len(trainval_df) * (1.0 - val_ratio))
    train_df = trainval_df.iloc[:inner_train_end]
    val_df = trainval_df.iloc[inner_train_end:]
    return train_df, val_df


def evaluate_long_horizon_rmse(
    model: PANORAMA,
    eval_df: pd.DataFrame,
    dt: float,
    device: torch.device,
    rollout_steps: int | None = None,
) -> float:
    theta, omega = get_theta_omega(eval_df, fallback_dt=dt)
    max_rollout_steps = len(theta) - 1
    if max_rollout_steps <= 0:
        raise ValueError("Validation split is too short for long-horizon rollout.")

    actual_steps = max_rollout_steps if rollout_steps is None else min(
        rollout_steps, max_rollout_steps
    )
    init_state = torch.tensor([[theta[0], omega[0]]], dtype=torch.float32, device=device)
    true_future = theta[1 : 1 + actual_steps]

    model.eval()
    with torch.no_grad():
        pred_traj, _ = model(init_state, actual_steps)
        pred_theta = pred_traj.squeeze(0).cpu().numpy()[:, 0]

    return calculate_rmse_numpy(pred_theta, true_future)


def run_trial(
    trial: optuna.Trial,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    seq_len_seconds: float,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    batch_size: int,
) -> float:
    device = torch.device(config["system"]["device"])
    fps = config["system"]["fps"]
    dt = 1.0 / fps
    seq_len = resolve_seq_len(seq_len_seconds, fps)

    torch.manual_seed(config["system"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["system"]["seed"])

    tune_cfg = config.get("tune", {})
    validation_rollout_seconds = tune_cfg.get("validation_rollout_seconds")
    validation_rollout_steps = None
    if validation_rollout_seconds is not None:
        validation_rollout_steps = round(validation_rollout_seconds * fps)

    train_loader = DataLoader(
        ODEDataset(train_df, seq_len=seq_len, dt=dt),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    supervision_indices = resolve_supervision_indices(
        seq_len,
        source_fps=fps,
        supervision_fps=config["train"].get("supervision_downsample_to_fps"),
    )

    model = PANORAMA(
        dt=dt,
        g=config["physics"]["g"],
        m=config["physics"]["m"],
        L=config["physics"]["L"],
        k1=config["physics"]["k1"],
        k2=config["physics"]["k2"],
        hidden_dim=hidden_dim,
        input_scale=config["model"]["input_scale"],
        residual_scale=config["model"].get("residual_scale", 1.0),
        output_init_std=config["model"].get("output_init_std", 1e-3),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    theta_loss_weight = config["train"].get("theta_loss_weight", 1.0)
    omega_loss_weight = config["train"].get("omega_loss_weight", 0.5)
    search_epochs = tune_cfg.get("search_epochs", 20)

    val_rmse = float("inf")
    for epoch in range(search_epochs):
        model.train()
        for init_state, target_traj in train_loader:
            init_state = init_state.to(device)
            target_traj = target_traj.to(device)

            optimizer.zero_grad()
            pred_traj, _ = model(init_state, seq_len)
            pred_traj, target_traj = apply_sparse_supervision(
                pred_traj,
                target_traj,
                supervision_indices,
            )
            theta_loss = calculate_mse(pred_traj[:, :, 0], target_traj[:, :, 0])
            omega_loss = calculate_mse(pred_traj[:, :, 1], target_traj[:, :, 1])
            loss = theta_loss_weight * theta_loss + omega_loss_weight * omega_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_rmse = evaluate_long_horizon_rmse(
            model=model,
            eval_df=val_df,
            dt=dt,
            device=device,
            rollout_steps=validation_rollout_steps,
        )
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_rmse


def objective_stage1(
    trial: optuna.Trial, config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> float:
    stage1_cfg = config.get("tune", {}).get("stage1", {})
    candidates = stage1_cfg.get(
        "seq_len_seconds_candidates", [1.0, 1.5, 2.0, 3.0, 4.0]
    )
    seq_len_seconds = trial.suggest_categorical("seq_len_seconds", candidates)
    return run_trial(
        trial=trial,
        config=config,
        train_df=train_df,
        val_df=val_df,
        seq_len_seconds=seq_len_seconds,
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        hidden_dim=config["model"]["hidden_dim"],
        batch_size=config["train"]["batch_size"],
    )


def objective_stage2(
    trial: optuna.Trial,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    seq_len_seconds: float,
) -> float:
    stage2_cfg = config.get("tune", {}).get("stage2", {})
    lr_min, lr_max = stage2_cfg.get("lr_range", [1e-4, 5e-3])
    wd_min, wd_max = stage2_cfg.get("weight_decay_range", [1e-6, 1e-3])
    hidden_choices = stage2_cfg.get("hidden_dim_choices", [32, 64, 128])
    batch_choices = stage2_cfg.get("batch_size_choices", [128, 256, 512])

    lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
    weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", hidden_choices)
    batch_size = trial.suggest_categorical("batch_size", batch_choices)

    return run_trial(
        trial=trial,
        config=config,
        train_df=train_df,
        val_df=val_df,
        seq_len_seconds=seq_len_seconds,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
    )


def main():
    config = load_config()
    tune_cfg = config.get("tune", {})
    stage1_cfg = tune_cfg.get("stage1", {})
    stage2_cfg = tune_cfg.get("stage2", {})
    train_df, val_df = get_train_val_split(config)

    print(f"Start two-stage Optuna search on dataset: {config['data']['active_dataset']}")

    stage1_study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    stage1_study.optimize(
        lambda trial: objective_stage1(trial, config, train_df, val_df),
        n_trials=stage1_cfg.get("n_trials", 8),
    )
    best_seq_len_seconds = stage1_study.best_params["seq_len_seconds"]

    print("\nStage 1 complete")
    print(f"Best seq_len_seconds: {best_seq_len_seconds}")
    print(f"Best long-horizon RMSE: {stage1_study.best_value:.8e}")

    stage2_study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    stage2_study.optimize(
        lambda trial: objective_stage2(
            trial,
            config,
            train_df,
            val_df,
            seq_len_seconds=best_seq_len_seconds,
        ),
        n_trials=stage2_cfg.get("n_trials", 20),
    )

    print("\nStage 2 complete")
    print(f"Best long-horizon RMSE: {stage2_study.best_value:.8e}")
    print("Recommended config updates:")
    print(f"  train.seq_len_seconds: {best_seq_len_seconds}")
    for key, value in stage2_study.best_params.items():
        if key == "hidden_dim":
            print(f"  model.hidden_dim: {value}")
        elif key == "batch_size":
            print(f"  train.batch_size: {value}")
        else:
            print(f"  train.{key}: {value}")


if __name__ == "__main__":
    main()
