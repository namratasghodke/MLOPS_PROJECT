import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Load raw dataset
df = pd.read_csv('../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')  


# Step 1: Basic Cleaning
df = df.dropna()  # or use df.fillna() if preferred
df = df.drop_duplicates()


# Optional: Remove unwanted columns
# df = df.drop(columns=['unnecessary_column'])

# Step 2: Identify features and target
target = 'Churn'  # Replace with your target column
X = df.drop(columns=[target])
y = df[target]


# Step 3: Identify column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Step 4: Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Step 5: Preprocess features
X_preprocessed = preprocessor.fit_transform(X)

# Step 6: Train/Validation/Test Split
# First split into train and temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_preprocessed, y, test_size=0.3, random_state=42, stratify=y
)

# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Now you have: X_train, y_train, X_val, y_val, X_test, y_test




# Create processed data directory if it doesn't exist
os.makedirs('../../data/processed', exist_ok=True)

# Save preprocessed full dataset (if needed)
df_processed = pd.DataFrame(X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed)
df_processed[target] = y.values

df_processed.to_csv('../../data/processed/preprocessed_data.csv', index=False)
