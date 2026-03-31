#!/usr/bin/env python
"""
Phase 3 Verification Script - Tests that all components are properly integrated
without running full training loops.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all Phase 3 dependencies can be imported."""
    print("\n🔍 Testing Phase 3 Imports...")
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import torchdiffeq
        print("  ✓ torchdiffeq")
    except ImportError as e:
        print(f"  ✗ torchdiffeq: {e}")
        return False
    
    try:
        import torch_geometric
        print("  ✓ torch_geometric")
    except ImportError as e:
        print(f"  ✗ torch_geometric: {e}")
        return False
    
    try:
        import shap
        print("  ✓ shap")
    except ImportError as e:
        print(f"  ✗ shap: {e}")
        return False
    
    return True


def test_model_classes():
    """Test that model classes can be instantiated."""
    print("\n🏗️  Testing Model Class Instantiation...")
    try:
        from ml.models.neural_ode_v2 import NeuralODEModel, NeuralODEConfig
        config = NeuralODEConfig()
        model = NeuralODEModel(config)
        print(f"  ✓ NeuralODEModel instantiated ({model.__class__.__name__})")
    except Exception as e:
        print(f"  ✗ NeuralODEModel: {e}")
        return False
    
    try:
        from ml.models.temporal_gat_v2 import TemporalGATModel, TemporalGATConfig
        config = TemporalGATConfig(num_nodes=180)
        model = TemporalGATModel(config)
        print(f"  ✓ TemporalGATModel instantiated ({model.__class__.__name__})")
    except Exception as e:
        print(f"  ✗ TemporalGATModel: {e}")
        return False
    
    return True


def test_data_availability():
    """Test that processed data files exist."""
    print("\n📊 Testing Data Availability...")
    data_root = Path("data/processed")
    files_needed = [
        "features_daily.csv",
        "timeseries_daily.csv",
        "graph_snapshot.csv"
    ]
    
    all_exist = True
    for fname in files_needed:
        fpath = data_root / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {fname} (missing)")
            all_exist = False
    
    return all_exist


def test_scenario_runner():
    """Test that scenario runner can be initialized."""
    print("\n🎯 Testing Scenario Runner...")
    try:
        from ml.inference.scenario_runner import ScenarioRunner
        runner = ScenarioRunner()
        print(f"  ✓ ScenarioRunner instantiated ({runner.__class__.__name__})")
        
        import numpy as np
        context = np.array([0.6, 0.8, 0.3, 0.1], dtype=np.float32)
        print(f"  ✓ Context shape OK: {context.shape}")
        
    except Exception as e:
        print(f"  ✗ ScenarioRunner: {e}")
        return False
    
    return True


def test_backend_services():
    """Test that backend services can load module setup."""
    print("\n🔌 Testing Backend Service Initialization...")
    try:
        from backend.app.services.forecast_service import _load_trained_models
        ode_model, gat_model = _load_trained_models()
        print(f"  ✓ _load_trained_models() callable")
        if ode_model is None:
            print(f"    (Note: ODE model not found in artifacts yet - will use fallback)")
        if gat_model is None:
            print(f"    (Note: GAT model not found in artifacts yet - will use fallback)")
    except Exception as e:
        print(f"  ✗ forecast_service: {e}")
        return False
    
    try:
        from backend.app.services.scenario_service import _get_scenario_runner
        runner = _get_scenario_runner()
        print(f"  ✓ _get_scenario_runner() callable")
    except Exception as e:
        print(f"  ✗ scenario_service: {e}")
        return False
    
    return True


def test_training_modules():
    """Test that training modules can be imported."""
    print("\n🎓 Testing Training Module Imports...")
    try:
        from ml.training.train_neural_ode_v2 import main
        print(f"  ✓ train_neural_ode_v2.main() importable")
    except Exception as e:
        print(f"  ✗ train_neural_ode_v2: {e}")
        return False
    
    try:
        from ml.training.train_temporal_gat_v2 import main
        print(f"  ✓ train_temporal_gat_v2.main() importable")
    except Exception as e:
        print(f"  ✗ train_temporal_gat_v2: {e}")
        return False
    
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("  PHASE 3 VERIFICATION SUITE")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_imports),
        ("Model Instantiation", test_model_classes),
        ("Data Availability", test_data_availability),
        ("Scenario Engine", test_scenario_runner),
        ("Backend Services", test_backend_services),
        ("Training Modules", test_training_modules),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("\n✅ ALL PHASE 3 COMPONENTS VERIFIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Run: python -m ml.training.train_neural_ode_v2")
        print("  2. Run: python -m ml.training.train_temporal_gat_v2")
        print("  3. Start API: python -m uvicorn backend.app.main:app --reload")
        print("  4. Open frontend: Open frontend/index.html in browser")
        return 0
    else:
        print("\n❌ SOME VERIFICATION TESTS FAILED")
        print("Check errors above and re-run after fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
