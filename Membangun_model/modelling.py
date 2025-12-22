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

# ======================================================
# 1. Set MLflow Experiment
# ======================================================
mlflow.set_tracking_uri(
    "sqlite:///D:/GUNADARMA UNIVERSITY/Asah Dicoding/Project/Deployment/"
    "MSML_Fajar-Agus-Dwi-Rahmawan/Membangun_model/mlflow.db"
)
mlflow.set_experiment("BMW_Price_Prediction_Basic")

def main():
    # ======================================================
    # 2. Load Preprocessed Dataset
    # ======================================================
    BASE_DIR = Path(__file__).parent
    df = pd.read_csv(BASE_DIR / "bmw_preprocessed.csv")

    X = df.drop(columns="price")
    y = df["price"]

    # ======================================================
    # 3. Train-Test Split
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ======================================================
    # 4. Enable MLflow Autolog
    # ======================================================
    mlflow.autolog()

    # ======================================================
    # 5. Model Training
    # ======================================================
    with mlflow.start_run(run_name="RandomForest_Basic_Model"):
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # ==================================================
        # 6. Model Evaluation
        # ==================================================
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MAPE
        mape = (abs((y_test - y_pred) / y_test)).mean() * 100

        # ==================================================
        # 7. Print Evaluation Metrics
        # ==================================================
        print("\n===== MODEL EVALUATION METRICS =====")
        print(f"MSE  : {mse:,.2f}")
        print(f"RMSE : {rmse:,.2f}")
        print(f"MAE  : {mae:,.2f}")
        print(f"R2   : {r2:.4f}")
        print(f"MAPE : {mape:.2f}%")

        # ==================================================
        # 8. Interpretation with Simple Threshold
        # ==================================================
        print("\nINTERPRETATION:")
        if r2 >= 0.9:
            print("✓ Excellent! Model explains >90% of price variance")
        elif r2 >= 0.8:
            print("✓ Very Good! Model explains >80% of price variance")
        elif r2 >= 0.7:
            print("✓ Good! Model explains >70% of price variance")
        else:
            print("⚠ Fair. Model could be improved with feature engineering or tuning")

        print(f"✓ On average, predictions are off by ${mae:,.2f}")
        print(f"✓ Prediction error is approximately {mape:.1f}% of actual price")

        # ==================================================
        # 9. Sample Prediction Comparison
        # ==================================================
        print("\n📋 SAMPLE PREDICTIONS (First 10 Test Samples)")
        print("-" * 85)
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

        print("-" * 85)


if __name__ == "__main__":
    main()
