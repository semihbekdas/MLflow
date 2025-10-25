import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

from src.data.make_dataset import prepare_datasets
from src.utils.io import ensure_dir, read_yaml

DEFAULT_CONFIG = "configs/config_titanic.yaml"


def load_test(cfg):
    p = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    if (p / "X_test.csv").exists():
        X_test = pd.read_csv(p / "X_test.csv")
        y_test = pd.read_csv(p / "y_test.csv").squeeze("columns")
        return X_test, y_test

    _, X_test, _, y_test = prepare_datasets(cfg)
    return X_test, y_test


def main():
    cfg = read_yaml(DEFAULT_CONFIG)
    mlflow.set_tracking_uri(cfg.get("tracking_uri", "file:./mlruns"))
    experiment_name = cfg.get("experiment_name", "mlflow_egitimi")
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    #Bu satır, mevcut experiment’in tüm bilgilerini (id, artifact_location, lifecycle_stage) çeker.

    if not exp:
        raise RuntimeError("Experiment not found. Run training first.")
    #Eğer hiçbir “mlflow_egitimi” experiment’i bulunmazsa hata verir

    client = MlflowClient()
    #mlflow_egitimi experiment’inde, tag’i train olan run’lar arasında en son yapılanı getir.
    filter_string = "tags.run_scope = 'train'"
    runs = client.search_runs(
        [exp.experiment_id], # 1️⃣ Hangi experiment’te arayacağız?
        filter_string=filter_string, # 2️⃣ Yukarıda tanımladığımız filtre
        order_by=["attributes.start_time DESC"], # 3️⃣ En son yapılan run en başta gelsin
        max_results=1,  # 4️⃣ Sadece en yenisini getir
    )
    
    if not runs:
        raise RuntimeError("No runs found. Train a model first.")

    run = runs[0]
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)
    X_test, y_test = load_test(cfg)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    with mlflow.start_run(run_name="evaluate_last_run", tags={"run_scope": "evaluation"}):
        mlflow.log_param("evaluated_run_id", run_id)
        mlflow.log_metric("eval_accuracy", float(acc))
        mlflow.log_metric("eval_f1_macro", float(f1))

        # Save sample predictions
        ensure_dir("artifacts")
        preview_count = int(cfg.get("evaluate", {}).get("top_n_samples_preview", 5))
        preview_df = X_test.head(preview_count).copy()
        preview_df["actual"] = y_test.head(preview_count).values
        preview_df["prediction"] = y_pred[:preview_count]
        path = os.path.join("artifacts", "prediction_preview.csv")
        preview_df.to_csv(path, index=False)
        mlflow.log_artifact(path)

    print(f"Evaluated run {run_id}: acc={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()
