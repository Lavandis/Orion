from .data import resolve_seq_len


def resolve_stage_epochs(total_epochs: int, stage_epoch_ratios: list[float]) -> list[int]:
    """Allocate an integer epoch budget across stages while preserving the total."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if not stage_epoch_ratios:
        raise ValueError("stage_epoch_ratios must not be empty.")
    if any(ratio <= 0 for ratio in stage_epoch_ratios):
        raise ValueError("stage_epoch_ratios must all be positive.")
    if total_epochs < len(stage_epoch_ratios):
        return [1 if idx < total_epochs else 0 for idx in range(len(stage_epoch_ratios))]

    total_ratio = sum(stage_epoch_ratios)
    stage_epochs: list[int] = []
    allocated = 0
    cumulative_target = 0.0

    for idx, ratio in enumerate(stage_epoch_ratios):
        remaining_stages = len(stage_epoch_ratios) - idx - 1
        if idx == len(stage_epoch_ratios) - 1:
            epochs = total_epochs - allocated
        else:
            cumulative_target += total_epochs * ratio / total_ratio
            target_so_far = round(cumulative_target)
            epochs = max(1, target_so_far - allocated)
            max_epochs = total_epochs - allocated - remaining_stages
            epochs = min(epochs, max_epochs)

        stage_epochs.append(epochs)
        allocated += epochs

    return stage_epochs


def get_curriculum_template_names(train_cfg: dict) -> list[str]:
    curriculum_cfg = train_cfg.get("curriculum", {})
    templates = curriculum_cfg.get("templates", {})
    return list(templates.keys())


def resolve_training_stages(
    train_cfg: dict,
    fps: float,
    *,
    total_epochs: int | None = None,
    selected_template: str | None = None,
) -> list[dict]:
    """Resolve either a fixed-window plan or a curriculum plan into stage metadata."""
    total_epochs = train_cfg["epochs"] if total_epochs is None else total_epochs
    curriculum_cfg = train_cfg.get("curriculum", {})

    if not curriculum_cfg.get("enabled", False):
        seq_len_seconds = train_cfg["seq_len_seconds"]
        return [
            {
                "name": "single_stage",
                "seq_len_seconds": seq_len_seconds,
                "seq_len": resolve_seq_len(seq_len_seconds, fps),
                "epochs": total_epochs,
            }
        ]

    templates = curriculum_cfg.get("templates", {})
    if not templates:
        raise ValueError("Curriculum is enabled, but no templates are configured.")

    template_name = (
        selected_template
        if selected_template is not None
        else curriculum_cfg.get("selected_template")
    )
    if template_name not in templates:
        raise ValueError(f"Unknown curriculum template: {template_name}")

    template_cfg = templates[template_name]
    stage_seq_len_seconds = template_cfg.get("stage_seq_len_seconds")
    if not stage_seq_len_seconds:
        raise ValueError(
            f"Curriculum template '{template_name}' is missing stage_seq_len_seconds."
        )

    stage_epoch_ratios = template_cfg.get(
        "stage_epoch_ratios",
        curriculum_cfg.get("stage_epoch_ratios", [0.2, 0.3, 0.5]),
    )
    if len(stage_epoch_ratios) != len(stage_seq_len_seconds):
        raise ValueError(
            f"Curriculum template '{template_name}' has mismatched stage lengths and ratios."
        )

    stage_epochs = resolve_stage_epochs(total_epochs, stage_epoch_ratios)
    stages = []
    for idx, (seq_len_seconds, epochs) in enumerate(
        zip(stage_seq_len_seconds, stage_epochs, strict=True)
    ):
        if epochs <= 0:
            continue
        stages.append(
            {
                "name": f"{template_name}_stage_{idx + 1}",
                "template_name": template_name,
                "seq_len_seconds": seq_len_seconds,
                "seq_len": resolve_seq_len(seq_len_seconds, fps),
                "epochs": epochs,
            }
        )

    return stages
