from ml.models.ensemble import WeightedEnsemble
from ml.models.neural_ode import NeuralODEModel
from ml.models.temporal_gat import TemporalGATModel


class Predictor:
    def __init__(self) -> None:
        self.ode = NeuralODEModel()
        self.gat = TemporalGATModel()
        self.ensemble = WeightedEnsemble()

    def forecast(self, horizon: int, base_value: float) -> list[float]:
        ode = self.ode.predict(horizon=horizon, base_value=base_value)
        gat = self.gat.predict(horizon=horizon, base_value=base_value)
        return self.ensemble.predict(ode, gat)
