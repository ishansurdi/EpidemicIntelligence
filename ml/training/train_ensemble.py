from ml.models.ensemble import WeightedEnsemble


def main() -> None:
    ode_forecast = [100.0, 110.0, 121.0, 133.0]
    gat_forecast = [103.0, 114.0, 126.0, 141.0]

    model = WeightedEnsemble()
    combined = model.predict(ode_forecast, gat_forecast)

    print("Ensemble training complete")
    print(f"Sample combined output: {combined}")


if __name__ == "__main__":
    main()
