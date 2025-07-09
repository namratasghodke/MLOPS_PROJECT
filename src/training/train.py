import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === CONFIGURATION ===
DATA_PATH = os.path.join('data/processed', 'preprocessed_data.csv')
TARGET_COLUMN = 'Churn'
EXPERIMENT_NAME = 'churn-prediction'
MODEL_NAME = 'Churn_Model'
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# === LOAD PROCESSED DATA ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Processed dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# === Identify column types ===
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# === Preprocessing pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# === Final ML pipeline ===
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(**params))
])

# === Split Train/Val ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Set MLflow Tracking URI and Experiment ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# === Start MLflow Run ===
with mlflow.start_run():
    # --- Train Model ---
    clf.fit(X_train, y_train)

    # --- Predict & Evaluate ---
    y_pred = clf.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred, pos_label='Yes')
    }

    # --- Log Parameters & Metrics ---
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # --- Save & Log Model ---
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    # --- Save Preprocessor for Inference Use ---
    preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    mlflow.log_artifact(preprocessor_path)

    # --- Log Confusion Matrix as Artifact ---
    cm = confusion_matrix(y_val, y_pred, labels=['No', 'Yes'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    print(f"\n✅ Model '{MODEL_NAME}' logged under experiment '{EXPERIMENT_NAME}'")
    print(f"📊 Metrics: {metrics}")
    print(f"📁 Preprocessing and confusion matrix artifacts saved.\n")
