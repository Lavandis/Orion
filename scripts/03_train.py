import os
import sys

import pandas as pd
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PANORAMA
from src.trainer import train_panorama
from src.utils import resolve_training_stages


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

    train_stages = resolve_training_stages(config["train"], fps)
    final_stage = train_stages[-1]
    config["train"]["seq_len"] = final_stage["seq_len"]

    curriculum_cfg = config["train"].get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        print(
            f"Curriculum template: {curriculum_cfg['selected_template']} | "
            f"stages={len(train_stages)}"
        )
        for idx, stage in enumerate(train_stages, start=1):
            print(
                f"  Stage {idx}: {stage['seq_len_seconds']:.2f}s -> "
                f"{stage['seq_len']} steps | epochs={stage['epochs']}"
            )
    else:
        print(
            f"Training horizon: {final_stage['seq_len_seconds']:.2f}s -> "
            f"{final_stage['seq_len']} steps"
        )

    supervision_fps = config["train"].get("supervision_downsample_to_fps")
    if supervision_fps is None:
        print("Training supervision: full-resolution")
    else:
        print(f"Training supervision: sparse @ {supervision_fps} fps")

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
        train_df=train_df,
        train_stages=train_stages,
        config=config,
        device=device,
        dt=dt,
    )
    print("==================================================")
    print("Training finished")


if __name__ == "__main__":
    main()
