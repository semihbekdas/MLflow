import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir, read_yaml


DEFAULT_CONFIG = "configs/config_titanic.yaml"


def _load_titanic_dataframe(csv_path: str) -> pd.DataFrame:
    """Kaggle Titanic verisini yükler ve kolon isimlerini normalize eder."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Titanic CSV bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    required = {"survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}. Lütfen Kaggle Titanic 'train.csv' dosyasını kullanın.")

    return df


def _process_titanic_features(df: pd.DataFrame, stats: Dict[str, Any] | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Titanic veri seti için feature engineering uygular ve train-istatistikleri ile eksikleri doldurur."""
    processed = df.copy()

    if stats is None:
        stats = {}
        stats["age_median"] = float(processed["age"].median(skipna=True))
        stats["fare_median"] = float(processed["fare"].median(skipna=True))
        embarked_mode_series = processed["embarked"].dropna()
        stats["embarked_mode"] = embarked_mode_series.mode().iloc[0] if not embarked_mode_series.empty else "S"
        stats["sex_fallback"] = processed["sex"].dropna().mode().iloc[0] if not processed["sex"].dropna().empty else "male"

    processed["age"] = processed["age"].fillna(stats["age_median"])
    processed["fare"] = processed["fare"].fillna(stats["fare_median"])
    processed["embarked"] = processed["embarked"].fillna(stats["embarked_mode"])
    processed["sex"] = processed["sex"].fillna(stats["sex_fallback"])

    processed["sex"] = processed["sex"].str.lower().str.strip()
    processed["embarked"] = processed["embarked"].str.upper().str.strip()

    sex_map = {"male": 0, "female": 1}
    processed["sex_encoded"] = processed["sex"].map(sex_map).fillna(sex_map.get(stats.get("sex_fallback", "male"), 0)).astype(int)

    embarked_map = {"S": 0, "C": 1, "Q": 2}
    fallback_embarked = embarked_map.get(stats.get("embarked_mode", "S"), 0)
    processed["embarked_encoded"] = (
        processed["embarked"].map(embarked_map).fillna(fallback_embarked).astype(int)
    )

    processed["family_size"] = processed["sibsp"].astype(int) + processed["parch"].astype(int) + 1
    processed["is_alone"] = (processed["family_size"] == 1).astype(int)

    bins = [-np.inf, 16, 30, 50, np.inf]
    labels = [0, 1, 2, 3]
    processed["age_group_encoded"] = pd.cut(processed["age"], bins=bins, labels=labels).astype(int)

    feature_cols = [
        "pclass",
        "sex_encoded",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked_encoded",
        "family_size",
        "is_alone",
        "age_group_encoded",
    ]

    return processed[feature_cols].copy(), stats


def _load_generic_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("CSV dosyasında 'target' kolonu bulunmalı.")
    X = df.drop(columns=["target"]).copy()
    y = df["target"].copy()
    return X, y


def prepare_datasets(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    processed_path = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    ensure_dir(processed_path)

    source = cfg.get("data", {}).get("source", "titanic").lower()
    if source == "titanic":
        csv_path = cfg.get("data", {}).get("csv_path", "data/raw/titanic.csv")
        raw_df = _load_titanic_dataframe(csv_path)
        y = raw_df["survived"].astype(int).rename("target")
        feature_source = raw_df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
    elif source == "csv":
        csv_path = cfg.get("data", {}).get("csv_path", "data/raw/data.csv")
        X, y = _load_generic_csv(csv_path)
    else:
        raise ValueError("data.source yalnızca 'titanic' ya da 'csv' olabilir.")

    test_size = float(cfg.get("data", {}).get("test_size", 0.2))
    stratify = bool(cfg.get("data", {}).get("stratify", True))
    random_state = int(cfg.get("random_state", 42))
    strat = y if stratify else None

    if source == "titanic":
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            feature_source,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
        X_train, stats = _process_titanic_features(X_train_raw)
        X_test, _ = _process_titanic_features(X_test_raw, stats)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

    X_train.to_csv(processed_path / "X_train.csv", index=False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Titanic verisini işleyip train/test dosyaları üretir.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Yapılandırma dosyası yolu")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    processed_path = cfg.get("data", {}).get("processed_dir", "data/processed")
    prepare_datasets(cfg)
    print(f"İşlenmiş veri kaydedildi: {processed_path}")


if __name__ == "__main__":
    main()
