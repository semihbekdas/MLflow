import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.make_dataset import prepare_datasets
from src.utils.io import ensure_dir, merge_overrides, read_yaml
from mlflow.models.signature import infer_signature


DEFAULT_CONFIG = "configs/config_titanic.yaml"


def load_processed_or_raw(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed")
    p = Path(processed_dir)

    if (p / "X_train.csv").exists():
        X_train = pd.read_csv(p / "X_train.csv")
        X_test = pd.read_csv(p / "X_test.csv")
        y_train = pd.read_csv(p / "y_train.csv").squeeze("columns")
        y_test = pd.read_csv(p / "y_test.csv").squeeze("columns")
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = prepare_datasets(cfg)
    return X_train, X_test, y_train, y_test


def build_model(cfg: Dict) -> Tuple[Pipeline, Dict[str, Any]]:
    model_name = cfg.get("model", {}).get("name", "RandomForestClassifier")
    params = cfg.get("model", {}).get("params", {})

    if model_name == "RandomForestClassifier":
        allowed = {"n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf", "bootstrap", "random_state"}
        rf_params = {k: v for k, v in params.items() if k in allowed and v is not None}
        model = RandomForestClassifier(**rf_params)
        steps = [("model", model)]
        used_params = rf_params
    elif model_name == "LogisticRegression":
        allowed = {"penalty", "C", "solver", "max_iter", "random_state", "multi_class"}
        lr_params = {k: v for k, v in params.items() if k in allowed and v is not None}
        lr_params.setdefault("max_iter", 2000)
        lr_params.setdefault("random_state", cfg.get("random_state", 42))
        model = LogisticRegression(**lr_params)
        steps = [("scaler", StandardScaler()), ("model", model)]
        used_params = lr_params
    elif model_name == "KNeighborsClassifier":
        allowed = {"n_neighbors", "weights", "metric"}
        knn_params = {k: v for k, v in params.items() if k in allowed and v is not None}
        model = KNeighborsClassifier(**knn_params)
        steps = [("scaler", StandardScaler()), ("model", model)]
        used_params = knn_params
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps), used_params


def train(cfg: Dict, run_name: str | None = None):
    tracking_uri = cfg.get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("experiment_name", "mlflow_egitimi"))

    autolog = bool(cfg.get("train", {}).get("autolog", False))
    if autolog:
        mlflow.sklearn.autolog(log_models=False)

    X_train, X_test, y_train, y_test = load_processed_or_raw(cfg)
    pipe, used_params = build_model(cfg)

    base_tags = {"run_scope": "train"}
    with mlflow.start_run(
        run_name=run_name or cfg.get("train", {}).get("run_name"),
        tags=base_tags,
    ):
        # Params and tags
        mlflow.log_param("model_name", cfg.get("model", {}).get("name"))
        for k, v in used_params.items():
            mlflow.log_param(f"model.{k}", v)

        for tk, tv in (cfg.get("train", {}).get("tags", {}) or {}).items():
            mlflow.set_tag(tk, tv)

        # Train
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        # Metrics
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))
        mlflow.log_metric("train_time_sec", float(train_time))

        # Confusion matrix artifact (optional)
        if cfg.get("evaluate", {}).get("save_confusion_matrix", True):
            import matplotlib.pyplot as plt

            cm = confusion_matrix(y_test, y_pred)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct = np.divide(cm, row_sums, where=row_sums != 0)
            class_labels = [str(cls) for cls in sorted(pd.unique(y_test))]

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_pct, cmap="Blues")
            ax.set_title("Confusion Matrix — Normalize (Row %)")
            ax.set_xlabel("Tahmin")
            ax.set_ylabel("Gerçek")
            ax.set_xticks(np.arange(len(class_labels)))
            ax.set_yticks(np.arange(len(class_labels)))
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    pct = cm_pct[i, j] * 100 if row_sums[i, 0] else 0
                    text = f"{count}\n({pct:0.1f}%)"
                    ax.text(j, i, text, ha="center", va="center", color="black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            ensure_dir("artifacts")
            fig_path = os.path.join("artifacts", "confusion_matrix.png")
            fig.savefig(fig_path, dpi=150)
            mlflow.log_artifact(fig_path)
            plt.close(fig)

        # Log model
        signature = infer_signature(X_train, pipe.predict(X_train))
        input_example = X_train.head(min(len(X_train), 5))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        rid = mlflow.active_run().info.run_id
        print(f"Run finished: run_id={rid} | acc={acc:.4f} | f1_macro={f1:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a classifier with MLflow logging")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--override", type=str, default="{}", help="JSON string to override config values")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg = merge_overrides(cfg, args.override)
    train(cfg, run_name=args.run_name)
