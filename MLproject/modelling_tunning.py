import os
import sys
import time
import json
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("file:./mlruns")

if len(sys.argv) < 2:
    raise ValueError("Please provide path to dataset as an argument.")

dataset_path = sys.argv[1]

print(f"[INFO] Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

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
    with mlflow.start_run() as run:
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
        recall = recall_score(y_test, y_pred, average='weighted')
        true_positive_rate = recall_score(y_test, y_pred, pos_label=1)
        false_positive_rate = 1 - precision_score(y_test, y_pred, pos_label=1)
        
        mlflow.log_metric({
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': prec,
            'recall': rec,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'training_time': end_time - start_time
        })
        
        mlflow.xgboost.log_model(xgb, artifact_path="model")
        
        ConfusionMatrixDisplay.from_estimator(
            xgb, X_test, y_test, cmap='Blues', normalize='true',
            display_labels=xgb.classes_, ax=None, colorbar=False
        )
        plt.savefig("confusion_matrix.png")
        
        with open("metric_info.json", "w") as f:
            json.dump({
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': prec,
                'recall': rec,
                'true_positive_rate': true_positive_rate,
                'false_positive_rate': false_positive_rate,
                'training_time': end_time - start_time
            }, f)
            
        
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("metric_info.json")
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_run_id = run_id
            print(f"[INFO] New best F1 score: {best_f1_score} for run {best_run_id}")
            
if os.path.exists("best_run_id.txt"):
    os.remove("best_run_id.txt")
    
with open("best_run_id.txt", "w") as f:
    f.write(best_run_id)
        