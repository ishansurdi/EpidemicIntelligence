from dataclasses import dataclass


@dataclass
class TemporalGATConfig:
    gat_layers: int = 3
    attention_heads: int = 8
    hidden_dim: int = 128
    transformer_layers: int = 4
    epochs: int = 200


class TemporalGATModel:
    def __init__(self, config: TemporalGATConfig | None = None) -> None:
        self.config = config or TemporalGATConfig()
        self.is_trained = False

    def fit(self, node_features, edge_index) -> None:
        # Phase 1 placeholder for PyG-based full-batch training.
        _ = (node_features, edge_index)
        self.is_trained = True

    def predict(self, horizon: int, base_value: float) -> list[float]:
        return [base_value * (1.0 + day * 0.025) for day in range(horizon)]

    def attention_edges(self) -> list[dict[str, str | float]]:
        return [
            {"source": "IND", "target": "ARE", "weight": 0.79},
            {"source": "USA", "target": "MEX", "weight": 0.71},
        ]
