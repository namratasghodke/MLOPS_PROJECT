import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
DATA_PATH = '../../data/processed/preprocessed_data.csv'
TARGET_COLUMN = 'Churn'
EXPERIMENT_NAME = 'churn-prediction'
MODEL_NAME = 'Churn_Model'

# === LOAD PROCESSED DATA ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# === SPLIT TRAIN/VAL ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Set Tracking URI and Experiment ===
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment(EXPERIMENT_NAME)

# === START RUN ===
with mlflow.start_run():
    # === Train Model ===
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # === Evaluate Model ===
    y_pred = model.predict(X_val)
    
    # 🔁 Change this if needed (e.g. pos_label=1 for binary numeric)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred, pos_label='Yes')
    }

    # === Log with MLflow ===
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model, artifact_path="model", registered_model_name=MODEL_NAME
    )

    print(f" Model logged and registered as '{MODEL_NAME}' under experiment '{EXPERIMENT_NAME}'")
    print(f" Metrics: {metrics}")


