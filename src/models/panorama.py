import torch
import torch.nn as nn

from .augmentation import AugmentationNetwork
from .physics import PhysicsBox
from src.utils.integrators import rk4_step


class PANORAMA(nn.Module):
    """Hybrid physics-neural dynamics model for a damped pendulum."""

    def __init__(
        self,
        dt: float,
        g: float,
        m: float,
        L: float,
        k1: float,
        k2: float,
        hidden_dim: int,
        input_scale: list[float],
        residual_scale: float = 1.0,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        self.dt = dt
        self.physics_model = PhysicsBox(g, m, L, k1, k2)
        self.augmentation = AugmentationNetwork(
            hidden_dim=hidden_dim,
            input_scale=input_scale,
            residual_scale=residual_scale,
            output_init_std=output_init_std,
        )

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        physics_term = self.physics_model(state)
        residual_term = self.augmentation(state)

        zeros = torch.zeros_like(residual_term)
        residual_vec = torch.cat([zeros, residual_term], dim=1)
        return physics_term + residual_vec

    def forward(
        self, initial_state: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = []
        fa_penalties = []
        curr_state = initial_state

        for _ in range(steps):
            aug_force = self.augmentation(curr_state)
            fa_penalties.append((aug_force.square()).mean())

            curr_state = rk4_step(self.dynamics, curr_state, self.dt)
            states.append(curr_state)

        pred_traj = torch.stack(states, dim=1)
        fa_penalty_mean = torch.stack(fa_penalties).mean()
        return pred_traj, fa_penalty_mean
