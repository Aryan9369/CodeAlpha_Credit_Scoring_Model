import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, auc)
# Optional: For handling imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn pipeline if using SMOTE

# --- 1. Load Data ---
# df = pd.read_csv('your_credit_data.csv')
# X = df.drop('target_variable_column_name', axis=1)
# y = df['target_variable_column_name']

# --- Placeholder Data (Replace with actual data loading) ---
# Example: Create dummy data for structure demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, weights=[0.9, 0.1], # Imbalanced
                           random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
# Add some simulated categorical features
X['cat_feature_1'] = pd.cut(X['feature_0'], bins=3, labels=['Low', 'Medium', 'High'])
X['cat_feature_2'] = pd.cut(X['feature_1'], bins=2, labels=['TypeA', 'TypeB'])
# Introduce some missing values
import numpy as np
X.iloc[::10, 0] = np.nan
X.iloc[::20, 1] = np.nan
X.iloc[::30, -1] = np.nan # Missing categorical

# --- Identify feature types ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# --- 2. Preprocessing ---
# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Handle missing numerical
    ('scaler', StandardScaler()) # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3 & 4. Model Training & Validation ---
# Split data (BEFORE any fitting/transformation leaks data from test to train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify for imbalance

# --- Choose a Model (Example: Random Forest) ---

# Define the model pipeline (including preprocessing and classifier)
# Use ImbPipeline if using SMOTE to ensure SMOTE is only applied during training folds
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    # ('smote', SMOTE(random_state=42)), # Optional: Add SMOTE for imbalance
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')) # Use class_weight for imbalance
])

# --- Hyperparameter Tuning (Example using GridSearchCV) ---
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1) # Score based on AUC
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score on CV: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# --- 5. Model Evaluation (on Test Set) ---
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1] # Probabilities for ROC AUC

print("\n--- Test Set Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}") # Use probabilities for AUC
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Good Risk (0)', 'Bad Risk (1)']))

# Plot ROC Curve (Optional)
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
