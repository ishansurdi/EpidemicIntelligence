"""
Scenario runner for computing counterfactual forecasts.
Allows intervention scenarios (mobility reduction, vaccination acceleration)
and recomputes predictions with learned models.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd

from ml.models.neural_ode_v2 import NeuralODEModel, NeuralODEConfig
from ml.models.temporal_gat_v2 import TemporalGATModel, TemporalGATConfig


class ScenarioRunner:
    """Runs counterfactual forecast scenarios with learned models."""

    def __init__(
        self,
        neural_ode_model: NeuralODEModel | None = None,
        temporal_gat_model: TemporalGATModel | None = None,
    ):
        self.ode_model = neural_ode_model
        self.gat_model = temporal_gat_model
        self.device = torch.device("cpu")

    @classmethod
    def from_artifacts(cls, artifact_root: Path) -> "ScenarioRunner":
        """Load trained models from artifact directory."""
        neural_ode_model = None
        temporal_gat_model = None

        try:
            ode_path = artifact_root / "neural_ode_model.pt"
            if ode_path.exists():
                config = NeuralODEConfig()
                neural_ode_model = NeuralODEModel(config)
                neural_ode_model.load_state_dict(torch.load(ode_path, map_location="cpu"))
                neural_ode_model.eval()
        except Exception as e:
            print(f"Warning: Could not load Neural ODE: {e}")

        try:
            gat_path = artifact_root / "temporal_gat_model.pt"
            if gat_path.exists():
                config = TemporalGATConfig(num_nodes=180)
                temporal_gat_model = TemporalGATModel(config)
                temporal_gat_model.load_state_dict(torch.load(gat_path, map_location="cpu"))
                temporal_gat_model.eval()
        except Exception as e:
            print(f"Warning: Could not load Temporal GAT: {e}")

        return cls(neural_ode_model, temporal_gat_model)

    def forecast_baseline(
        self, context: np.ndarray, horizon: int = 14
    ) -> np.ndarray:
        """Run baseline forecast without interventions.

        Args:
            context: (4,) context features [vaccination, mobility, stringency, acceleration]
            horizon: number of days to forecast

        Returns:
            (horizon,) predicted daily cases
        """
        if self.ode_model is None:
            return np.zeros(horizon)

        forecasts = []
        current_context = context.copy()

        for _ in range(horizon):
            context_tensor = torch.from_numpy(current_context).float().unsqueeze(0)
            y0 = torch.tensor([[0.99, 0.005, 0.005, 0.0]], dtype=torch.float32)
            t_span = torch.linspace(0, 1.0, 8, dtype=torch.float32)

            try:
                with torch.no_grad():
                    solution = self.ode_model(y0, t_span, context_tensor)
                    pred_infected = float(solution[-1, 0, 2].numpy())
                    pred_cases = max(pred_infected * 1500.0, 0.0)
                    forecasts.append(pred_cases)
            except Exception:
                forecasts.append(0.0)

            current_context[3] = max(current_context[3] - 0.01, 0.0)

        return np.array(forecasts)

    def forecast_with_intervention(
        self,
        context: np.ndarray,
        intervention_type: str = "mobility_reduction",
        intervention_strength: float = 0.2,
        horizon: int = 14,
    ) -> np.ndarray:
        """Run forecast with a counterfactual intervention.

        Args:
            context: baseline context features
            intervention_type: 'mobility_reduction' or 'vaccination_acceleration'
            intervention_strength: magnitude of intervention (0-1)
            horizon: forecast days

        Returns:
            (horizon,) predicted daily cases under intervention
        """
        if self.ode_model is None:
            return np.zeros(horizon)

        forecasts = []
        current_context = context.copy()

        for day in range(horizon):
            if intervention_type == "mobility_reduction":
                modified_context = current_context.copy()
                modified_context[1] = max(
                    modified_context[1] - intervention_strength * 0.05, 0.0
                )
            elif intervention_type == "vaccination_acceleration":
                modified_context = current_context.copy()
                modified_context[0] = min(
                    modified_context[0] + intervention_strength * 0.02, 1.0
                )
            else:
                modified_context = current_context.copy()

            context_tensor = torch.from_numpy(modified_context).float().unsqueeze(0)
            y0 = torch.tensor([[0.99, 0.005, 0.005, 0.0]], dtype=torch.float32)
            t_span = torch.linspace(0, 1.0, 8, dtype=torch.float32)

            try:
                with torch.no_grad():
                    solution = self.ode_model(y0, t_span, context_tensor)
                    pred_infected = float(solution[-1, 0, 2].numpy())
                    pred_cases = max(pred_infected * 1500.0, 0.0)
                    forecasts.append(pred_cases)
            except Exception:
                forecasts.append(0.0)

            current_context = modified_context.copy()
            current_context[3] = max(current_context[3] - 0.01, 0.0)

        return np.array(forecasts)

    def compare_scenarios(
        self,
        context: np.ndarray,
        scenarios: Dict[str, Tuple[str, float]],
        horizon: int = 14,
    ) -> Dict[str, np.ndarray]:
        """Compare multiple intervention scenarios.

        Args:
            context: baseline context
            scenarios: Dict[name, (intervention_type, strength)]
            horizon: forecast days

        Returns:
            Dict[scenario_name, forecasts]
        """
        results = {}

        baseline = self.forecast_baseline(context, horizon)
        results["baseline"] = baseline

        for scenario_name, (intervention_type, strength) in scenarios.items():
            try:
                forecast = self.forecast_with_intervention(
                    context, intervention_type, strength, horizon
                )
                results[scenario_name] = forecast
            except Exception as e:
                print(f"Scenario {scenario_name} failed: {e}")
                results[scenario_name] = None

        return results

    def compute_impact(
        self, baseline: np.ndarray, intervention: np.ndarray
    ) -> Dict[str, float]:
        """Compute intervention impact metrics.

        Args:
            baseline: baseline forecast
            intervention: counterfactual forecast

        Returns:
            Dict with impact metrics
        """
        if intervention is None or len(intervention) == 0:
            return {}

        total_baseline = baseline.sum()
        total_intervention = intervention.sum()
        averted = total_baseline - total_intervention
        pct_reduction = (averted / max(total_baseline, 1.0)) * 100.0

        return {
            "total_baseline_cases": float(total_baseline),
            "total_intervention_cases": float(total_intervention),
            "averted_cases": float(averted),
            "percent_reduction": float(pct_reduction),
        }
