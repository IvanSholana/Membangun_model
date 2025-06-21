import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

# === Load data ===
dataset_df = pd.read_csv("./personality_preprocessing/personality_preprocessing.csv")

x_train, x_test, y_train, y_test = train_test_split(
    dataset_df.drop(columns=["Personality"]),
    dataset_df["Personality"],
    test_size=0.2,
    random_state=42
)

# === MLflow setup ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Personality_Prediction_Experiment")

# Enable MLflow autologging for XGBoost
mlflow.xgboost.autolog()

# === Training ===
if __name__ == "__main__":
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    print("Training XGBoost model with MLflow autologging...")
    with mlflow.start_run(run_name="XGBoost"):
        start_time = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start_time

        # Manual log: training time (autolog doesn't do this)
        mlflow.log_metric("train_time", train_time)

        print(f"[✓] Training completed in {train_time:.2f} seconds.")