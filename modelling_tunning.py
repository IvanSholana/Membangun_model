from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.xgboost
import time
import pandas as pd
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Personality_Prediction_Experiment")

dataset = pd.read_csv("./Membangun_model/personality_preprocessing/personality_preprocessing.csv")
x_train, x_test, y_train, y_test = train_test_split(
    dataset.drop(columns=["Personality"]),
    dataset["Personality"],
    test_size=0.2,
    random_state=42
)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

with mlflow.start_run(run_name="XGB_Tunning") as parent_run:
    grid_seach = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_seach.fit(x_train, y_train)
    train_time = time.time() - start_time
    
    for i in range(len(grid_seach.cv_results_['params'])):
        params = grid_seach.cv_results_['params'][i]
        
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        with mlflow.start_run(run_name=f"XGB_Tunning_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average='binary'))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average='binary'))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='binary'))
            mlflow.log_metric("train_time", train_time)
            mlflow.log_metric("true_positive_rate", recall_score(y_test, y_pred, pos_label=1))
            mlflow.log_metric("true_negative_rate", recall_score(y_test, y_pred, pos_label=0))
            mlflow.xgboost.log_model(model, "model")
            
    y_pred_best = grid_seach.best_estimator_.predict(x_test)
    best_params = grid_seach.best_params_
    best_model = grid_seach.best_estimator_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", accuracy_score(y_test, y_pred_best))
    mlflow.log_metric("best_precision", precision_score(y_test, y_pred_best, average='binary'))
    mlflow.log_metric("best_recall", recall_score(y_test, y_pred_best, average='binary'))
    mlflow.log_metric("best_f1_score", f1_score(y_test, y_pred_best, average='binary'))
    mlflow.log_metric("best_train_time", train_time)
    mlflow.xgboost.log_model(best_model, "best_model")
    
    