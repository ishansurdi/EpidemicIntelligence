from ml.data.feature_engine import add_temporal_features, melt_jhu_confirmed
from ml.data.loaders import load_jhu_confirmed
from ml.models.neural_ode import NeuralODEModel


def main() -> None:
    jhu = load_jhu_confirmed()
    features = add_temporal_features(melt_jhu_confirmed(jhu))

    model = NeuralODEModel()
    model.fit(features)

    print("Neural ODE training complete")
    print(f"Rows used: {len(features)}")


if __name__ == "__main__":
    main()
