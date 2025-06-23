#!/usr/bin/env python3
"""
Script alternatif untuk training tanpa MLflow nested runs
yang bisa menyebabkan error "Run not found"
"""

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

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
print("🚀 Starting GridSearch training...")
start = time.time()
grid.fit(X_train, y_train)
total_train_time = time.time() - start
print(f"⏱️  Training completed in {total_train_time:.2f} seconds")

# ─────────────── Evaluate All Candidates ───────────────
print("📊 Evaluating all candidates...")
all_results = []

for idx, params in enumerate(grid.cv_results_["params"]):
    model = XGBClassifier(**params, eval_metric="mlogloss") 
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "candidate_id": idx,
        "parameters": params,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro"),
        "recall": recall_score(y_test, preds, average="macro"),
        "f1_score": f1_score(y_test, preds, average="macro"),
        "cv_score": grid.cv_results_["mean_test_score"][idx],
        "cv_std": grid.cv_results_["std_test_score"][idx]
    }
    all_results.append(metrics)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_filename = f"confusion_matrix_{idx}.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Candidate {idx} Confusion Matrix\nAccuracy: {metrics['accuracy']:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_filename)
    plt.close()
    
    print(f"  Candidate {idx}: CV={metrics['cv_score']:.4f}, Test Acc={metrics['accuracy']:.4f}")

# ─────────────── Best Model Results ───────────────
best_model = grid.best_estimator_
best_params = grid.best_params_
best_preds = best_model.predict(X_test)

best_metrics = {
    "best_parameters": best_params,
    "accuracy": accuracy_score(y_test, best_preds),
    "precision": precision_score(y_test, best_preds, average="macro"),
    "recall": recall_score(y_test, best_preds, average="macro"),
    "f1_score": f1_score(y_test, best_preds, average="macro"),
    "train_time": total_train_time,
    "best_cv_score": grid.best_score_
}

# Plot confusion matrix for best model
best_cm_path = "training_confusion_matrix.png"
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, fmt="d", cmap="Blues")
plt.title(f"Best Model Confusion Matrix\nAccuracy: {best_metrics['accuracy']:.4f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(best_cm_path)
plt.close()

# ─────────────── Save Results ───────────────
# Save all results
results_json = "all_candidates_results.json"
with open(results_json, "w") as f:
    json.dump(all_results, f, indent=4)

# Save best metrics
best_json = "metric_info.json"
with open(best_json, "w") as f:
    json.dump(best_metrics, f, indent=4)

# Save model info
best_html = "estimator.html"
with open(best_html, "w") as f:
    f.write(f"<h2>Best XGBoost Model</h2><pre>{best_model}</pre>")
    f.write(f"<h3>Best Parameters:</h3><pre>{json.dumps(best_params, indent=2)}</pre>")
    f.write(f"<h3>Performance Metrics:</h3><pre>{json.dumps(best_metrics, indent=2)}</pre>")

# ─────────────── MLflow Logging (Simple) ───────────────
try:
    import mlflow
    import mlflow.xgboost
    
    # Simple MLflow logging without nested runs
    mlflow.set_tracking_uri("file:./mlruns")
    
    with mlflow.start_run(run_name="XGBoost_GridSearch_Simple"):
        # Log best parameters
        mlflow.log_params(best_params)
        
        # Log best metrics
        for k, v in best_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        
        # Log model
        mlflow.xgboost.log_model(best_model, artifact_path="best_model")
        
        # Log artifacts
        mlflow.log_artifact(best_cm_path)
        mlflow.log_artifact(best_json)
        mlflow.log_artifact(best_html)
        mlflow.log_artifact(results_json)
        
        print("✅ MLflow logging completed successfully!")
        
except Exception as e:
    print(f"⚠️  MLflow logging failed: {e}")
    print("📁 Results saved to local files anyway")

# ─────────────── Final Summary ───────────────
print("\n" + "="*60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"🏆 Best Parameters: {best_params}")
print(f"📊 Best CV Score: {best_metrics['best_cv_score']:.4f}")
print(f"📈 Test Accuracy: {best_metrics['accuracy']:.4f}")
print(f"📈 Test Precision: {best_metrics['precision']:.4f}")
print(f"📈 Test Recall: {best_metrics['recall']:.4f}")
print(f"📈 Test F1-Score: {best_metrics['f1_score']:.4f}")
print(f"⏱️  Total Training Time: {total_train_time:.2f} seconds")
print("\n📁 Output Files:")
print(f"  - {best_cm_path}")
print(f"  - {best_json}")
print(f"  - {best_html}")
print(f"  - {results_json}")
print("="*60)
