import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ml.models.neural_ode import NeuralODEConfig


class SEIRDynamics(nn.Module):
    """Physics-informed ODE system with learned time-varying transmission rates."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Context encoder: maps covariates to rate parameters
        # Inputs: vaccination rate, mobility index, policy stringency, 7d case velocity
        self.context_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Rate heads: output transmission (β), recovery (γ), exposed-to-infectious (σ)
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, t, y, context):
        batch_size = y.shape[0]
        hidden = self.context_encoder(context)
        
        beta = 0.5 * self.beta_head(hidden).view(batch_size)
        gamma = 0.1 * self.gamma_head(hidden).view(batch_size)
        sigma = 0.2 * self.sigma_head(hidden).view(batch_size)
        
        S, E, I, R = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        N = S + E + I + R + 1e-8
        
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        
        return torch.stack([dS, dE, dI, dR], dim=1)


class NeuralODEModel(nn.Module):
    """Neural ODE forecaster with learned SEIR dynamics."""

    def __init__(self, config: NeuralODEConfig | None = None):
        super().__init__()
        self.config = config or NeuralODEConfig()
        self.dynamics = SEIRDynamics(hidden_dim=16)
        # Learnable output head: maps (SEIR state + context hidden) → normalized case prediction.
        # This ensures gradients flow through a linear path regardless of ODE solver.
        self.output_head = nn.Sequential(
            nn.Linear(4 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, y0: torch.Tensor, t_span: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        y0: (batch, 4) initial SEIR state
        t_span: (T,) time points
        context: (batch, 4) feature context
        Returns: (T, batch, 4) ODE solution trajectory
        """
        def ode_func(t, y):
            return self.dynamics(t, y, context)

        solution = odeint(ode_func, y0, t_span, method="dopri5", rtol=1e-3, atol=1e-4)
        return solution

    def forward_normalized(
        self, y0: torch.Tensor, t_span: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pred_normalized, solution) where pred_normalized is (batch,) in [0,1]
        via a learnable projection so gradient flows cleanly.
        """
        solution = self.forward(y0, t_span, context)           # (T, batch, 4)
        state_final = solution[-1]                              # (batch, 4)
        context_hidden = self.dynamics.context_encoder(context) # (batch, 16)
        combined = torch.cat([state_final, context_hidden], dim=-1)  # (batch, 20)
        pred = self.output_head(combined).squeeze(-1)           # (batch,)
        return pred, solution

    def predict(self, horizon: int, base_value: float) -> list[float]:
        y0 = torch.tensor([[0.99, 0.005, 0.005, 0.0]], dtype=torch.float32)
        t_span = torch.linspace(0, horizon / 7.0, horizon, dtype=torch.float32)
        context = torch.tensor([[0.5, 0.8, 0.5, 0.02]], dtype=torch.float32)

        with torch.no_grad():
            solution = self.forward(y0, t_span, context)

        infectious_trajectory = solution[:, 0, 2].numpy()
        scaled = base_value * (infectious_trajectory / (infectious_trajectory.max() + 1e-8))
        return [max(float(v), 0.0) for v in scaled]
