import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    # ======================================================
    # 1. Set MLflow Tracking to DagsHub
    # ======================================================
    DAGSHUB_USER = os.getenv("DAGSHUB_USER")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

    mlflow.set_tracking_uri(
        f"https://{DAGSHUB_USER}:{DAGSHUB_TOKEN}@dagshub.com/{DAGSHUB_USER}/BMW_Price_Prediction.mlflow"
    )

    mlflow.set_experiment("BMW_Price_Prediction_Advance")

    # ======================================================
    # 2. Load Dataset
    # ======================================================
    df = pd.read_csv(
        "bmw_preprocessed.csv"
    )

    X = df.drop(columns="price")
    y = df["price"]

    # ======================================================
    # 3. Train-Test Split
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================================================
    # 4. Hyperparameter Tuning
    # ======================================================
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5]
    }

    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    with mlflow.start_run(run_name="RandomForest_Advance"):

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # ==================================================
        # 5. Evaluation
        # ==================================================
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # ==================================================
        # 6. Manual Logging (PARAMS & METRICS)
        # ==================================================
        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ==================================================
        # 7. Log Model
        # ==================================================
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # ==================================================
        # 8. ADDITIONAL ARTIFACT 1 — Feature Importance
        # ==================================================
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        # ==================================================
        # 9. ADDITIONAL ARTIFACT 2 — Prediction Result
        # ==================================================
        prediction_result = pd.DataFrame({
            "actual_price": y_test.values,
            "predicted_price": y_pred,
            "error": y_pred - y_test.values
        })

        prediction_result.to_csv("prediction_result.csv", index=False)
        mlflow.log_artifact("prediction_result.csv")

        print("ADVANCE RUN COMPLETED SUCCESSFULLY")
        print("R2:", r2)


if __name__ == "__main__":
    main()
