import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.xgboost
import sys

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ─────────────── MLflow Setup ───────────────
try:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("personality-model-tuning")
    print("✅ MLflow tracking berhasil diinisialisasi")
except Exception as e:
    print(f"⚠️  Warning MLflow setup: {e}")
    # Fallback ke tracking lokal
    mlflow.set_tracking_uri("file:./mlruns")

# ───────────────────────── Read dataset ─────────────────────────
if len(sys.argv) < 2:
    raise ValueError("Please provide path to dataset as an argument.")
dataset_path = sys.argv[1]
print(f"[INFO] Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

X = df.drop(columns=["Personality"])
y = df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────── Grid Search Setup ───────────────
param_grid = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.1, 0.2],
}
grid = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1
)

# ─────────────── Fit Grid ───────────────
start = time.time()
grid.fit(X_train, y_train)
total_train_time = time.time() - start

# ─────────────── Start Parent Run ───────────────
with mlflow.start_run(run_name="XGBoost_GridSearch_Tuning") as parent_run:
    # Log parent run info
    mlflow.log_param("total_candidates", len(grid.cv_results_["params"]))
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("scoring", "accuracy")
    
    # ─────────────── Log Each Candidate as Nested Run ───────────────
    for idx, params in enumerate(grid.cv_results_["params"]):
        model = XGBClassifier(**params, eval_metric="mlogloss") 
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="macro"),
            "recall": recall_score(y_test, preds, average="macro"),
            "f1_score": f1_score(y_test, preds, average="macro"),
            "train_time": total_train_time
        }

        # Plot confusion matrix
        cm = confusion_matrix(y_test, preds)
        cm_filename = f"confusion_matrix_{idx}.png"
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Tuning {idx} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(cm_filename)
        plt.close()

        # Save metrics JSON
        metric_path = f"metric_info_{idx}.json"
        with open(metric_path, "w") as fp:
            json.dump(metrics, fp, indent=4)

        # Save estimator info
        estimator_html = f"estimator_{idx}.html"
        with open(estimator_html, "w") as fp:
            fp.write(f"<pre>{model}</pre>")

        # Log to MLflow as nested run
        with mlflow.start_run(run_name=f"XGB_Candidate_{idx}", nested=True):
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.xgboost.log_model(model, artifact_path="model")
            mlflow.log_artifact(cm_filename)
            mlflow.log_artifact(metric_path)
            mlflow.log_artifact(estimator_html)

        # Optional: clean up local files after logging
        os.remove(cm_filename)
        os.remove(metric_path)
        os.remove(estimator_html)

    # ─────────────── Log Best Model to the parent run ───────────────
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_preds = best_model.predict(X_test)

    best_metrics = {
        "accuracy": accuracy_score(y_test, best_preds),
        "precision": precision_score(y_test, best_preds, average="macro"),
        "recall": recall_score(y_test, best_preds, average="macro"),
        "f1_score": f1_score(y_test, best_preds, average="macro"),
        "train_time": total_train_time
    }

    # Plot confusion matrix for best model
    best_cm_path = "training_confusion_matrix.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(best_cm_path)
    plt.close()

    # Save best metrics & model summary
    best_json = "metric_info.json"
    best_html = "estimator.html"
    with open(best_json, "w") as f:
        json.dump(best_metrics, f, indent=4)
    with open(best_html, "w") as f:
        f.write(f"<pre>{best_model}</pre>")

    # Log best model to the parent run
    mlflow.log_params(best_params)
    for k, v in best_metrics.items():
        mlflow.log_metric(k, v)
    mlflow.xgboost.log_model(best_model, artifact_path="best_model")
    mlflow.log_artifact(best_cm_path)
    mlflow.log_artifact(best_json)
    mlflow.log_artifact(best_html)

    # Optional: clean up local files
    os.remove(best_cm_path)
    os.remove(best_json)
    os.remove(best_html)
    
    print(f"✅ Experiment completed successfully!")
    print(f"🔗 Parent run ID: {parent_run.info.run_id}")
    print(f"🏆 Best parameters: {best_params}")
    print(f"📊 Best accuracy: {best_metrics['accuracy']:.4f}")