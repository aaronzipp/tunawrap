XGB_PARAMS = {
    "n_estimators": {"type": "int", "low": 200, "high": 2000},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
    "max_depth": {"type": "int", "low": 2, "high": 12},
    "min_child_weight": {"type": "int", "low": 1, "high": 50},
    "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
}

LGBM_PARAMS = {
    "n_estimators": {"type": "int", "low": 200, "high": 2000},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
    "num_leaves": {"type": "int", "low": 15, "high": 512},
    "min_child_samples": {"type": "int", "low": 5, "high": 100},
    "feature_fraction": {"type": "float", "low": 0.4, "high": 1.0},
}
