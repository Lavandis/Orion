import os
import sys

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ODEDataset
from src.models import PANORAMA
from src.trainer import train_panorama
from src.utils import resolve_seq_len


def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = torch.device(config["system"]["device"])
    fps = config["system"]["fps"]
    dt = 1.0 / fps

    torch.manual_seed(config["system"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["system"]["seed"])

    print(f"System initialized | device={device} | fps={fps}")

    data_path = config["data"]["active_dataset"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find processed dataset: {data_path}\nRun scripts/01_preprocess.py first."
        )

    print(f"Loading dataset: {data_path}")
    full_df = pd.read_csv(data_path)

    train_ratio = config["data"].get("train_ratio", 0.75)
    train_size = int(len(full_df) * train_ratio)
    train_df = full_df.iloc[:train_size]

    seq_len_seconds = config["train"]["seq_len_seconds"]
    seq_len = resolve_seq_len(seq_len_seconds, fps)
    config["train"]["seq_len"] = seq_len

    batch_size = config["train"]["batch_size"]
    train_dataset = ODEDataset(train_df, seq_len=seq_len, dt=dt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    print(f"Training horizon: {seq_len_seconds:.2f}s -> {seq_len} steps")
    print("Building PANORAMA model...")
    model = PANORAMA(
        dt=dt,
        g=config["physics"]["g"],
        m=config["physics"]["m"],
        L=config["physics"]["L"],
        k1=config["physics"]["k1"],
        k2=config["physics"]["k2"],
        hidden_dim=config["model"]["hidden_dim"],
        input_scale=config["model"]["input_scale"],
        residual_scale=config["model"].get("residual_scale", 1.0),
        output_init_std=config["model"].get("output_init_std", 1e-3),
    ).to(device)

    print("==================================================")
    train_panorama(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
    )
    print("==================================================")
    print("Training finished")


if __name__ == "__main__":
    main()
