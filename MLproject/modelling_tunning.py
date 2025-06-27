import os
import sys
import time
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

import os
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

if len(sys.argv) < 2:
    raise ValueError("Please provide path to dataset as an argument.")

dataset_path = sys.argv[1]

try:
    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"[ERROR] Dataset file not found: {dataset_path}")
    sys.exit(1)

X = df.drop(columns=["Personality"])
y = df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
}

combinations = list(ParameterGrid(param_grid))

best_f1_score = 0
best_run_id = None

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

for i, params in enumerate(combinations, 1):
    with mlflow.start_run(nested=True) as run:
        if mlflow.active_run() is not None:
            print(f"[DEBUG] Active run ID: {mlflow.active_run().info.run_id}")
            
        run_id = run.info.run_id
        print(f"[INFO] Running combination {i}/{len(combinations)}: {params}")
        
        mlflow.log_params(params)
        
        start_time = time.time()
        xgb.set_params(**params)
        xgb.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred = xgb.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("training_time", end_time - start_time)
        
        mlflow.xgboost.log_model(xgb, artifact_path="model")
        
        ConfusionMatrixDisplay.from_estimator(
            xgb, X_test, y_test, cmap='Blues', normalize='true',
            display_labels=xgb.classes_, ax=None, colorbar=False
        )
        plt.savefig(f"confusion_matrix_{run_id}.png")
        plt.close()
        
        metrics = {
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': prec,
            'recall': rec,
            'training_time': end_time - start_time
        }
        try:
            with open(f"metric_info_{run_id}.json", "w") as f:
                json.dump(metrics, f)
            mlflow.log_artifact(f"confusion_matrix_{run_id}.png")
            mlflow.log_artifact(f"metric_info_{run_id}.json")
        except Exception as e:
            print(f"[ERROR] Failed to save artifacts: {e}")
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_run_id = run_id
            print(f"[INFO] New best F1 score: {best_f1_score} for run {best_run_id}")  

try:
    if os.path.exists("best_run_id.txt"):
        os.remove("best_run_id.txt")
    with open("best_run_id.txt", "w") as f:
        f.write(best_run_id)
except Exception as e:
    print(f"[ERROR] Failed to write best_run_id.txt: {e}")
    sys.exit(1)