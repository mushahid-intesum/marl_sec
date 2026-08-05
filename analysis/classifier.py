import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Optional
from collect.dataset import TimingDataset

TEST_SIZE = 0.2
RF_N_ESTIMATORS = 100
MLP_HIDDEN = (64, 32)
MLP_MAX_ITER = 500


def _prepare_features(dataset: TimingDataset, use_per_op: bool = True) -> np.ndarray:
    arrays = dataset.to_arrays()
    total = arrays["total_cycles"].astype(np.float64).reshape(-1, 1)
    if use_per_op:
        per_op = arrays["per_op_cycles"].astype(np.float64)
        active_cols = np.any(per_op > 0, axis=0)
        if np.any(active_cols):
            return np.hstack([total, per_op[:, active_cols]])
    return total


def train_timing_classifier(dataset: TimingDataset,
                            model_type: str = "rf",
                            seed: int = 42,
                            use_per_op: bool = True) -> Dict:
    X = _prepare_features(dataset, use_per_op)
    y = np.array(dataset.actions)

    n_classes = dataset.act_dim
    random_baseline = 1.0 / n_classes

    if len(np.unique(y)) < 2:
        return {
            "model": None,
            "model_type": model_type,
            "accuracy": 1.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "confusion_matrix": np.array([[len(y)]]),
            "random_baseline": random_baseline,
            "n_train": 0,
            "n_test": 0,
            "n_features": X.shape[1],
            "y_test": y,
            "y_pred": y,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                     random_state=seed, n_jobs=-1)
    elif model_type == "mlp":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=MLP_HIDDEN,
                                  max_iter=MLP_MAX_ITER,
                                  random_state=seed, early_stopping=True)),
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    return {
        "model": clf,
        "model_type": model_type,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "random_baseline": random_baseline,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X.shape[1],
        "y_test": y_test,
        "y_pred": y_pred,
    }


def evaluate_classifier(result: Dict) -> Dict[str, float]:
    acc = result["accuracy"]
    baseline = result["random_baseline"]
    lift = acc / baseline if baseline > 0 else 0.0
    return {
        "accuracy": acc,
        "random_baseline": baseline,
        "lift_over_random": lift,
        "f1_macro": result["f1_macro"],
        "f1_weighted": result["f1_weighted"],
        "above_random": acc > baseline,
    }
