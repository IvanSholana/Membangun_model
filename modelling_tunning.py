import os
import sys
import time
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("Personality_Prediction_Experiment")

# ───────────────────────── Read dataset ─────────────────────────
if len(sys.argv) < 2:
    raise ValueError("Please provide path to dataset as an argument.")
dataset_path = sys.argv[1]
print(f"[INFO] Loading dataset from: {dataset_path}")
dataset = pd.read_csv(dataset_path)

# ───────────────────────── Train / test split ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(columns=["Personality"]),
    dataset["Personality"],
    test_size=0.2,
    random_state=42
)

# ───────────────────────── Model & grid ─────────────────────────
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

# ──────────────────────── Main MLflow run ───────────────────────
with mlflow.start_run(run_name="XGB_Tuning", nested=mlflow.active_run() is not None) as parent_run:
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    # ── Log every candidate model as a nested run ──
    for i, params in enumerate(grid_search.cv_results_["params"]):
        model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="mlogloss"
        ).fit(X_train, y_train)

        y_pred = model.predict(X_test)

        with mlflow.start_run(run_name=f"XGB_Tuning_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("recall",    recall_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("train_time", train_time)
            mlflow.xgboost.log_model(model, artifact_path="model")

    # ── Log the best model in the parent run ──
    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred_best = best_model.predict(X_test)

    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy",  accuracy_score(y_test, y_pred_best))
    mlflow.log_metric("best_precision", precision_score(y_test, y_pred_best, average="macro"))
    mlflow.log_metric("best_recall",    recall_score(y_test, y_pred_best, average="macro"))
    mlflow.log_metric("best_f1_score",  f1_score(y_test, y_pred_best, average="macro"))
    mlflow.log_metric("best_train_time", train_time)

    mlflow.xgboost.log_model(best_model, artifact_path="best_model")
    mlflow.xgboost.save_model(best_model, "model")   # Optional: for Docker etc.
