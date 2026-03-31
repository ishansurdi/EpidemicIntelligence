from ml.data.feature_engine import add_temporal_features, melt_jhu_confirmed
from ml.data.graph_builder import build_country_graph
from ml.data.loaders import load_google_mobility, load_jhu_confirmed
from ml.models.temporal_gat import TemporalGATModel


def main() -> None:
    jhu = load_jhu_confirmed()
    _features = add_temporal_features(melt_jhu_confirmed(jhu))
    mobility = load_google_mobility()
    graph = build_country_graph(mobility)

    model = TemporalGATModel()
    model.fit(node_features=None, edge_index=graph)

    print("Temporal GAT training complete")
    print(f"Graph edges: {len(graph)}")


if __name__ == "__main__":
    main()
