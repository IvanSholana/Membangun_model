import os
import sys
import time
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.xgboost
import dagshub

# ============== LOAD ENV ==============
mlflow.set_tracking_uri("https://dagshub.com/IvanSholana/Membangun_model.mlflow")

# ============== DAGSHUB INIT ==============
dagshub.init(
    repo_owner="IvanSholana",
    repo_name="Membangun_model",
    mlflow=True,
)

mlflow.set_experiment("Personality_Prediction_Experiment")

# ============== READ DATASET ARG ==============
if len(sys.argv) < 2:
    raise ValueError("Please provide path to dataset as an argument.")
dataset_path = sys.argv[1]
print(f"[INFO] Loading dataset from: {dataset_path}")
dataset = pd.read_csv(dataset_path)

# ============== SPLIT DATA ==============
x_train, x_test, y_train, y_test = train_test_split(
    dataset.drop(columns=["Personality"]),
    dataset["Personality"],
    test_size=0.2,
    random_state=42
)

# ============== DEFINE MODEL ==============
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

# ============== MAIN RUN ==============
with mlflow.start_run(run_name="XGB_Tuning") as parent_run:
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    start_time = time.time()
    grid_search.fit(x_train, y_train)
    train_time = time.time() - start_time

    for i, params in enumerate(grid_search.cv_results_["params"]):
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        with mlflow.start_run(run_name=f"XGB_Tuning_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average="binary"))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average="binary"))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="binary"))
            mlflow.log_metric("train_time", train_time)
            mlflow.log_metric("true_positive_rate", recall_score(y_test, y_pred, pos_label=1))
            mlflow.log_metric("true_negative_rate", recall_score(y_test, y_pred, pos_label=0))
            mlflow.xgboost.log_model(model, "model")

    # ============== BEST MODEL LOGGING ==============
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred_best = best_model.predict(x_test)

    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", accuracy_score(y_test, y_pred_best))
    mlflow.log_metric("best_precision", precision_score(y_test, y_pred_best, average="binary"))
    mlflow.log_metric("best_recall", recall_score(y_test, y_pred_best, average="binary"))
    mlflow.log_metric("best_f1_score", f1_score(y_test, y_pred_best, average="binary"))
    mlflow.log_metric("best_train_time", train_time)

    mlflow.xgboost.log_model(best_model, "best_model")

    # Optional: Save to local folder 'model/' for Docker build
    mlflow.xgboost.save_model(best_model, "model")