from dataclasses import dataclass


@dataclass
class NeuralODEConfig:
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10


class NeuralODEModel:
    def __init__(self, config: NeuralODEConfig | None = None) -> None:
        self.config = config or NeuralODEConfig()
        self.is_trained = False

    def fit(self, train_frame) -> None:
        # Phase 1 placeholder for training loop integration with torchdiffeq.
        _ = train_frame
        self.is_trained = True

    def predict(self, horizon: int, base_value: float) -> list[float]:
        growth = []
        for day in range(horizon):
            growth.append(base_value * (1.0 + day * 0.02))
        return growth
