# CodeAlpha_Credit_Scoring_Model


## Project Overview

This repository contains a Python script demonstrating the development of a simple credit scoring model. The goal is to predict the creditworthiness of individuals (whether they are likely to default or repay a loan) based on simulated financial features. This is framed as a binary classification problem.

The script utilizes synthetic, imbalanced data generated using `sklearn.datasets.make_classification` for demonstration purposes. It walks through key steps of a machine learning workflow: data preprocessing, model training (using Random Forest), hyperparameter tuning, and evaluation.

## Workflow / Features

The script performs the following steps:

1.  **Data Generation:** Creates a synthetic dataset (`X`, `y`) with numerical and categorical features. The dataset is intentionally **imbalanced** (simulating typical credit default scenarios where defaults are rarer). Missing values are also introduced artificially.
2.  **Feature Identification:** Automatically separates numerical and categorical features.
3.  **Preprocessing Pipeline:**
    *   **Numerical Features:** Imputes missing values using the median and scales features using `StandardScaler`.
    *   **Categorical Features:** Imputes missing values using the most frequent category and encodes features using `OneHotEncoder`.
    *   Uses `ColumnTransformer` to apply these steps selectively.
4.  **Data Splitting:** Splits the data into training (80%) and testing (20%) sets, ensuring the class distribution is maintained (`stratify=y`).
5.  **Model Training & Pipeline:**
    *   Uses `imblearn.pipeline.Pipeline` to integrate preprocessing steps with the classifier.
    *   Chooses `RandomForestClassifier` as the classification algorithm.
    *   Addresses class imbalance using the `class_weight='balanced'` parameter within the classifier. (Note: SMOTE is included as a commented-out option within the pipeline for easy experimentation).
6.  **Hyperparameter Tuning:**
    *   Uses `GridSearchCV` with 5-fold cross-validation (`KFold`) on the training data to find the best hyperparameters for the `RandomForestClassifier` (`n_estimators`, `max_depth`, `min_samples_split`).
    *   Optimizes based on the `roc_auc` score, which is suitable for imbalanced datasets.
7.  **Evaluation:**
    *   Trains the final model using the best parameters found by `GridSearchCV` on the entire training set.
    *   Evaluates the model's performance on the **unseen test set** using various metrics:
        *   Accuracy
        *   ROC AUC Score
        *   Confusion Matrix
        *   Classification Report (Precision, Recall, F1-Score per class)
8.  **Visualization:** Plots the ROC (Receiver Operating Characteristic) curve for the test set predictions.

## Libraries Used

*   `pandas`: For data manipulation and DataFrame creation.
*   `numpy`: For numerical operations and handling NaN values.
*   `scikit-learn`: For machine learning tasks (data splitting, preprocessing, modeling, evaluation, hyperparameter tuning).
*   `imbalanced-learn`: Specifically for the `Pipeline` that correctly handles resampling (like SMOTE, though not enabled by default here) during cross-validation and the `SMOTE` algorithm itself (commented out).
*   `matplotlib`: For plotting the ROC curve.

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Make sure you have pip installed. You'll primarily need `scikit-learn` and `imbalanced-learn`, which will pull in other dependencies like `pandas` and `numpy`. Matplotlib is needed for the plot.
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib
    ```
    *(Alternatively, you could create a `requirements.txt` file listing these and run `pip install -r requirements.txt`)*

4.  **Execute the Script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

5.  **Running on Kaggle:**
    *   Create a new Kaggle Notebook.
    *   In the first code cell, run: `!pip install -U imbalanced-learn`
    *   Paste the rest of the script content into the next code cell.
    *   Run the cells.

## Expected Output

Running the script will produce the following output in your console:

1.  The best hyperparameters found by `GridSearchCV`.
2.  The best mean cross-validated ROC AUC score achieved during training.
3.  A divider (`--- Test Set Evaluation ---`).
4.  Performance metrics calculated on the test set:
    *   Accuracy score.
    *   ROC AUC score.
    *   Confusion Matrix.
    *   Classification Report (showing precision, recall, f1-score for both 'Good Risk' and 'Bad Risk' classes).
5.  A pop-up window (or inline plot in environments like Jupyter/Kaggle) displaying the ROC Curve with the AUC value.

## Potential Improvements / Next Steps

*   Replace synthetic data with a real-world credit dataset (e.g., from Kaggle competitions, UCI Machine Learning Repository).
*   Experiment with other classification algorithms (e.g., Logistic Regression, SVM, Gradient Boosting models like XGBoost, LightGBM).
*   Perform more extensive feature engineering based on domain knowledge.
*   Explore different techniques for handling class imbalance (e.g., uncommenting and tuning SMOTE, trying undersampling methods).
*   Implement model interpretability techniques (e.g., SHAP, LIME) to understand feature importance and prediction reasoning.
*   Deploy the trained model as an API for real-time predictions.

