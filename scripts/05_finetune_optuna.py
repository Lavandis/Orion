import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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


def load_config(config_path: str) -> dict:
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

    tune_cfg = config.get("fine_tune", {})
    val_ratio = tune_cfg.get(
        "validation_ratio",
        config.get("tune", {}).get("validation_ratio", 0.2),
    )
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


def local_seq_len_candidates(config: dict) -> list[float]:
    fine_cfg = config.get("fine_tune", {})
    configured = fine_cfg.get("seq_len_seconds_candidates")
    if configured:
        return sorted({float(value) for value in configured if float(value) > 0})

    center = float(config["train"]["seq_len_seconds"])
    multipliers = fine_cfg.get("seq_len_seconds_multipliers", [0.75, 1.0, 1.25])
    candidates = {round(center * float(multiplier), 4) for multiplier in multipliers}
    return sorted(value for value in candidates if value > 0)


def suggest_scaled_float(
    trial: optuna.Trial,
    name: str,
    center: float,
    multiplier_range: list[float],
    *,
    log: bool = True,
) -> float:
    if center <= 0:
        raise ValueError(f"{name} center must be positive for scaled fine tuning.")

    low_multiplier, high_multiplier = multiplier_range
    low = center * float(low_multiplier)
    high = center * float(high_multiplier)
    if low <= 0 or high <= 0:
        raise ValueError(f"{name} search bounds must be positive.")
    if low > high:
        low, high = high, low

    return trial.suggest_float(name, low, high, log=log)


def run_trial(
    trial: optuna.Trial,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    search_epochs: int,
    validation_rollout_seconds: float | None,
) -> float:
    fine_cfg = config.get("fine_tune", {})
    train_cfg = config["train"]
    model_cfg = config["model"]

    seq_len_seconds = trial.suggest_categorical(
        "seq_len_seconds", local_seq_len_candidates(config)
    )
    lr = suggest_scaled_float(
        trial,
        "lr",
        float(train_cfg["lr"]),
        fine_cfg.get("lr_multiplier_range", [0.5, 2.0]),
    )
    weight_decay = suggest_scaled_float(
        trial,
        "weight_decay",
        float(train_cfg["weight_decay"]),
        fine_cfg.get("weight_decay_multiplier_range", [0.33, 3.0]),
    )

    base_fa_reg = float(train_cfg.get("fa_reg_weight", 0.0))
    if base_fa_reg > 0:
        fa_reg_weight = suggest_scaled_float(
            trial,
            "fa_reg_weight",
            base_fa_reg,
            fine_cfg.get("fa_reg_weight_multiplier_range", [0.5, 2.0]),
        )
    else:
        fa_reg_weight = 0.0

    hidden_choices = fine_cfg.get("hidden_dim_choices")
    if hidden_choices:
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_choices)
    else:
        hidden_dim = model_cfg["hidden_dim"]

    batch_choices = fine_cfg.get("batch_size_choices")
    if batch_choices:
        batch_size = trial.suggest_categorical("batch_size", batch_choices)
    else:
        batch_size = train_cfg["batch_size"]

    device = torch.device(config["system"]["device"])
    fps = config["system"]["fps"]
    dt = 1.0 / fps
    seq_len = resolve_seq_len(seq_len_seconds, fps)

    torch.manual_seed(config["system"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["system"]["seed"])

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
        supervision_fps=train_cfg.get("supervision_downsample_to_fps"),
    )

    model = PANORAMA(
        dt=dt,
        g=config["physics"]["g"],
        m=config["physics"]["m"],
        L=config["physics"]["L"],
        k1=config["physics"]["k1"],
        k2=config["physics"]["k2"],
        hidden_dim=hidden_dim,
        input_scale=model_cfg["input_scale"],
        residual_scale=model_cfg.get("residual_scale", 1.0),
        output_init_std=model_cfg.get("output_init_std", 1e-3),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    theta_loss_weight = train_cfg.get("theta_loss_weight", 1.0)
    omega_loss_weight = train_cfg.get("omega_loss_weight", 0.5)
    warmup_epochs = config["multiplier_method"]["warmup_epochs"]
    validation_interval_epochs = max(1, fine_cfg.get("validation_interval_epochs", 1))

    val_rmse = float("inf")
    for epoch in range(search_epochs):
        model.train()
        fa_weight = 0.0 if epoch < warmup_epochs else 1.0
        for init_state, target_traj in train_loader:
            init_state = init_state.to(device)
            target_traj = target_traj.to(device)

            optimizer.zero_grad()
            pred_traj, fa_norm = model(init_state, seq_len)
            pred_traj, target_traj = apply_sparse_supervision(
                pred_traj,
                target_traj,
                supervision_indices,
            )
            theta_loss = calculate_mse(pred_traj[:, :, 0], target_traj[:, :, 0])
            omega_loss = calculate_mse(pred_traj[:, :, 1], target_traj[:, :, 1])
            data_loss = theta_loss_weight * theta_loss + omega_loss_weight * omega_loss
            loss = data_loss + fa_weight * fa_reg_weight * fa_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        is_validation_epoch = (epoch + 1) % validation_interval_epochs == 0
        is_last_epoch = epoch == search_epochs - 1
        if is_validation_epoch or is_last_epoch:
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

    trial.set_user_attr("seq_len_steps", seq_len)
    trial.set_user_attr("supervised_frames", len(supervision_indices))
    return val_rmse


def create_artifact_dir(config: dict, output_root: str) -> Path:
    dataset_name = Path(config["data"]["active_dataset"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / "optuna_finetune" / f"{timestamp}_{dataset_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_artifacts(study: optuna.Study, config: dict, run_dir: Path) -> None:
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    study.trials_dataframe().to_csv(run_dir / "trials.csv", index=False)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    with open(run_dir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best, f, sort_keys=False, allow_unicode=True)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Optuna fine tuning.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--validation-rollout-seconds", type=float, default=None)
    parser.add_argument("--output-root", default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    fine_cfg = config.get("fine_tune", {})

    n_trials = args.trials or fine_cfg.get("n_trials", 20)
    search_epochs = args.epochs or fine_cfg.get(
        "search_epochs",
        config.get("tune", {}).get("search_epochs", 20),
    )
    validation_rollout_seconds = (
        args.validation_rollout_seconds
        if args.validation_rollout_seconds is not None
        else fine_cfg.get(
            "validation_rollout_seconds",
            config.get("tune", {}).get("validation_rollout_seconds", 30.0),
        )
    )

    train_df, val_df = get_train_val_split(config)
    run_dir = create_artifact_dir(config, args.output_root)

    print(f"Start local Optuna fine tuning on dataset: {config['data']['active_dataset']}")
    print(f"Artifacts: {run_dir}")
    print(f"Center params:")
    print(f"  seq_len_seconds={config['train']['seq_len_seconds']}")
    print(f"  lr={config['train']['lr']}")
    print(f"  weight_decay={config['train']['weight_decay']}")
    print(f"  fa_reg_weight={config['train'].get('fa_reg_weight', 0.0)}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: run_trial(
            trial,
            config,
            train_df,
            val_df,
            search_epochs=search_epochs,
            validation_rollout_seconds=validation_rollout_seconds,
        ),
        n_trials=n_trials,
    )

    save_artifacts(study, config, run_dir)

    print("\nFine tuning complete")
    print(f"Best long-horizon RMSE: {study.best_value:.8e}")
    print("Recommended config updates:")
    for key, value in study.best_params.items():
        if key == "hidden_dim":
            print(f"  model.hidden_dim: {value}")
        elif key == "batch_size":
            print(f"  train.batch_size: {value}")
        else:
            print(f"  train.{key}: {value}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
