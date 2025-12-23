import pandas as pd
import mlflow
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from mlflow.tracking import MlflowClient

# ======================================================
# 1. Project & MLflow Configuration
# ======================================================
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlflow_artifacts"

ARTIFACT_ROOT.mkdir(exist_ok=True)
EXPERIMENT_NAME = "BMW_Price_Prediction_Basic"

# Set tracking URI (WAJIB sebelum apapun)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

# Init client
client = MlflowClient()

# Create experiment if not exists
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    exp_id = client.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=f"file:///{ARTIFACT_ROOT}"
    )
    print(f"✓ Experiment created: {EXPERIMENT_NAME}")
else:
    exp_id = experiment.experiment_id
    print(f"✓ Using existing experiment: {EXPERIMENT_NAME}")

mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 70)
print("MLflow Configuration")
print("=" * 70)
print(f"Project Root   : {PROJECT_ROOT}")
print(f"Tracking URI   : {mlflow.get_tracking_uri()}")
print(f"Artifact Store : {ARTIFACT_ROOT}")
print("=" * 70)

# ======================================================
# 2. Main Pipeline
# ======================================================
def main():
    BASE_DIR = Path(__file__).parent
    data_path = BASE_DIR / "bmw_preprocessed.csv"

    print("\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    print(f"✓ Dataset shape: {df.shape}")

    X = df.drop(columns="price")
    y = df["price"]

    print("\n[2/5] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================================================
    # 3. Enable Autologging
    # ======================================================
    mlflow.autolog(
        log_models=True,
        silent=False
    )

    print("\n[3/5] Training model...")
    with mlflow.start_run(run_name="RandomForest_Basic_Model") as run:

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        print(f"✓ Run ID        : {run.info.run_id}")
        print(f"✓ Experiment ID : {run.info.experiment_id}")

        print("\n[4/5] Evaluating model...")
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = (abs((y_test - y_pred) / y_test)).mean() * 100

        print("\nMODEL METRICS")
        print("=" * 50)
        print(f"MSE  : {mse:,.2f}")
        print(f"RMSE : {rmse:,.2f}")
        print(f"MAE  : {mae:,.2f}")
        print(f"R²   : {r2:.4f}")
        print(f"MAPE : {mape:.2f}%")
        print("=" * 50)

        print("\n[5/5] Saving run ID...")
        run_id_file = PROJECT_ROOT / "current_model_run_id.txt"
        run_id_file.write_text(run.info.run_id)

        print(f"✓ Run ID saved to: {run_id_file}")

        print("\nNEXT STEPS")
        print("=" * 70)
        print("1. MLflow UI:")
        print(f"   mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}")
        print("\n2. Serve model:")
        print(
            f"   mlflow models serve "
            f"--backend-store-uri sqlite:///{MLFLOW_DB} "
            f"--model-uri runs:/{run.info.run_id}/model -p 5001"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
