import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import zipfile

# Load the creditcard.csv dataset from the archive
zip_path = r"c:\Users\Jay\Downloads\archive (1).zip"
csv_file = "creditcard.csv"

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_file) as file:
        data = pd.read_csv(file)

# Data preprocessing
print("Dataset loaded with shape:", data.shape)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Identify categorical columns and encode them
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Separate features and target variable
target_column = 'Class'  # assuming 'Class' is the target (fraud = 1, not fraud = 0)
X = data.drop(columns=[target_column])  # Remove the target column
y = data[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalize numerical features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train and evaluate multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{name} ROC AUC Score: {roc_auc_score(y_test, y_pred):.4f}")