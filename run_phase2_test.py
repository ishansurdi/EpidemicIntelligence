"""
Quick test script to verify Phase 2 backend integration.
Run as: python run_phase2_test.py
"""

import json
import sys
from pathlib import Path

import requests


def test_api() -> None:
    api_base = "http://127.0.0.1:8000/api/v1"

    print("=== Phase 2 Backend Integration Test ===\n")

    tests = {
        "health": ("GET", f"{api_base}/health"),
        "data/timeseries": ("GET", f"{api_base}/data/timeseries?country=India&start_date=2023-01-01"),
        "data/features": ("GET", f"{api_base}/data/features"),
        "data/graph": ("GET", f"{api_base}/data/graph"),
        "predict/forecast": (
            "POST",
            f"{api_base}/predict/forecast",
            {"region_ids": ["IND", "BRA", "USA"], "horizon": 14},
        ),
        "predict/outbreak-risk": (
            "POST",
            f"{api_base}/predict/outbreak-risk",
            {"region_ids": ["IND", "BRA"]},
        ),
        "cascade/trace": ("GET", f"{api_base}/cascade/trace/IND"),
        "interpret/attention-map": ("GET", f"{api_base}/interpret/attention-map"),
        "interpret/feature-importance": ("GET", f"{api_base}/interpret/feature-importance/IND"),
    }

    for name, test_spec in tests.items():
        method = test_spec[0]
        url = test_spec[1]
        payload = test_spec[2] if len(test_spec) > 2 else None

        try:
            if method == "GET":
                resp = requests.get(url, timeout=5)
            else:
                resp = requests.post(url, json=payload, timeout=5)

            status = "✓" if resp.status_code == 200 else f"✗ ({resp.status_code})"
            print(f"{status} {name}")
            if resp.status_code == 200:
                data = resp.json()
                keys = list(data.keys())[:3]
                print(f"  Keys: {', '.join(keys)}")
        except Exception as e:
            print(f"✗ {name}: {e}")

    print("\nPhase 2 test complete.")


if __name__ == "__main__":
    test_api()
