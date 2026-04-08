import torch
import torch.nn as nn


class AugmentationNetwork(nn.Module):
    """Neural residual that corrects the angular-acceleration dynamics."""

    def __init__(
        self,
        hidden_dim: int,
        input_scale: list[float] | None = None,
        residual_scale: float = 1.0,
        output_init_std: float = 1e-3,
    ):
        super().__init__()

        if input_scale is None:
            input_scale = [10.0, 1.0]

        self.register_buffer(
            "input_scale", torch.tensor(input_scale, dtype=torch.float32)
        )
        self.residual_scale = residual_scale

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

        # A small random init keeps the branch close to the physics baseline
        # without completely blocking gradients to the hidden layers.
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=output_init_std)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        scaled_state = state * self.input_scale
        features = self.net(scaled_state)
        return self.output_layer(features) * self.residual_scale
