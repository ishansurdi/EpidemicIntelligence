def trace_cascade(region_id: str) -> dict[str, object]:
    chain = [
        {"region": "ARE", "lag_days": 4, "attention_weight": 0.63},
        {"region": "IND", "lag_days": 10, "attention_weight": 0.77},
    ]
    tree = {
        region_id: ["ARE"],
        "ARE": ["IND"],
        "IND": [],
    }
    return {"origin_chain": chain, "cascade_tree": tree}
