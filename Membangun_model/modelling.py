import pandas as pd
import mlflow
import numpy as np
import mlflow.sklearn
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
# 1. Set MLflow Tracking URI - KE ROOT PROJECT!
# ======================================================
# Get root project directory (naik 1 level dari Membangun_model)
client = MlflowClient()
experiment_name = "BMW_Price_Prediction_Basic"
existing_exp = client.get_experiment_by_name(experiment_name)


PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlflow_artifacts"

if existing_exp is None:
    exp_id = client.create_experiment(
        name=experiment_name,
        artifact_location=f"file:///{ARTIFACT_ROOT}"
    )
    print(f"✓ Experiment created: {experiment_name}")
else:
    exp_id = existing_exp.experiment_id
    print(f"✓ Using existing experiment: {experiment_name}")

# Set tracking URI
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

print("=" * 70)
print("MLflow Configuration")
print("=" * 70)
print(f"Project Root    : {PROJECT_ROOT}")
print(f"Database Path   : {MLFLOW_DB}")
print(f"Tracking URI    : {mlflow.get_tracking_uri()}")
print(f"Database Exists : {MLFLOW_DB.exists()}")
print("=" * 70)

# Set experiment
mlflow.set_experiment(experiment_name)

def main():
    # ======================================================
    # 2. Load Preprocessed Dataset
    # ======================================================
    BASE_DIR = Path(__file__).parent
    data_path = BASE_DIR / "bmw_preprocessed.csv"
    
    print(f"\n[1/6] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape}")

    X = df.drop(columns="price")
    y = df["price"]

    # ======================================================
    # 3. Train-Test Split
    # ======================================================
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train set: {X_train.shape}")
    print(f"✓ Test set : {X_test.shape}")

    # ======================================================
    # 4. Model Training with MLflow Tracking
    # ======================================================
    print("\n[3/6] Training model...")
    
    with mlflow.start_run(run_name="RandomForest_Basic_Model") as run:
        
        print(f"\n✓ Run ID: {run.info.run_id}")
        print(f"✓ Experiment ID: {run.info.experiment_id}")
        
        # Model parameters
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        print("✓ Model trained successfully")

        # ======================================================
        # 5. Model Evaluation
        # ======================================================
        print("\n[4/6] Evaluating model...")
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = (abs((y_test - y_pred) / y_test)).mean() * 100

        # ======================================================
        # 6. Log to MLflow
        # ======================================================
        print("\n[5/6] Logging to MLflow...")
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "mape": mape
        }
        mlflow.log_metrics(metrics)
        
        # Log model with signature
        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name="BMW_Price_Predictor"
        )
        
        print("✓ All data logged to MLflow")

        # ======================================================
        # 7. Print Evaluation Metrics
        # ======================================================
        print("\n" + "=" * 70)
        print("MODEL EVALUATION METRICS")
        print("=" * 70)
        print(f"MSE  : ${mse:,.2f}")
        print(f"RMSE : ${rmse:,.2f}")
        print(f"MAE  : ${mae:,.2f}")
        print(f"R²   : {r2:.4f}")
        print(f"MAPE : {mape:.2f}%")
        print("=" * 70)

        # ======================================================
        # 8. Interpretation
        # ======================================================
        print("\nINTERPRETATION:")
        if r2 >= 0.9:
            print("✓ Excellent! Model explains >90% of price variance")
        elif r2 >= 0.8:
            print("✓ Very Good! Model explains >80% of price variance")
        elif r2 >= 0.7:
            print("✓ Good! Model explains >70% of price variance")
        else:
            print("⚠ Fair. Model could be improved")

        print(f"✓ On average, predictions are off by ${mae:,.2f}")
        print(f"✓ Prediction error is approximately {mape:.1f}% of actual price")

        # ======================================================
        # 9. Sample Prediction Comparison
        # ======================================================
        print("\n[6/6] Sample Predictions")
        print("=" * 85)
        print(f"{'Actual Price':<20} {'Predicted Price':<20} {'Difference':<20} {'Error %'}")
        print("-" * 85)

        y_test_reset = y_test.reset_index(drop=True)

        for i in range(min(10, len(y_test_reset))):
            actual = y_test_reset.iloc[i]
            predicted = y_pred[i]
            diff = predicted - actual
            error_pct = abs(diff / actual) * 100

            print(
                f"${actual:>18,.2f} "
                f"${predicted:>18,.2f} "
                f"${diff:>18,.2f} "
                f"{error_pct:>8.2f}%"
            )

        print("=" * 85)
        
        # ======================================================
        # 10. Save Run ID untuk Reference
        # ======================================================
        run_id_file = PROJECT_ROOT / "current_model_run_id.txt"
        with open(run_id_file, "w") as f:
            f.write(run.info.run_id)
        
        print(f"\n✓ Run ID saved to: {run_id_file}")
        
        # ======================================================
        # 11. Instructions
        # ======================================================
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. View in MLflow UI:")
        print(f"   mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}")
        print("\n2. Serve model:")
        print(f"   cd {PROJECT_ROOT}")
        print(f"   mlflow models serve --model-uri runs:/{run.info.run_id}/model -p 5001")
        # print("\n3. Or use the helper script:")
        # print(f"   python serve_model.py")
        print("=" * 70)


if __name__ == "__main__":
    main()