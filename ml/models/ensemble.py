from dataclasses import dataclass


@dataclass
class EnsembleConfig:
    ode_weight: float = 0.45
    gat_weight: float = 0.55


class WeightedEnsemble:
    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()

    def predict(self, ode_forecast: list[float], gat_forecast: list[float]) -> list[float]:
        mixed: list[float] = []
        for ode_value, gat_value in zip(ode_forecast, gat_forecast):
            value = (self.config.ode_weight * ode_value) + (self.config.gat_weight * gat_value)
            mixed.append(value)
        return mixed
