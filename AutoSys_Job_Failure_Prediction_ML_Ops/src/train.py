# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import os
import joblib
import shutil
import datetime

# Visualization (optional)
import matplotlib.pyplot as plt
#import seaborn as sns

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===============================
# MAIN FUNCTION
# ===============================
def train_final_model():

    print("Step 1: Loading dataset...")

    # Use relative path (IMPORTANT for GitHub)
    df = pd.read_csv("data/autosys_dataset.csv")

    # ===============================
    # EDA
    # ===============================
    print("\nStep 2: EDA")
    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nTarget Distribution:\n", df['status'].value_counts())

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    print("\nStep 3: Encoding target")

    le = LabelEncoder()
    df['status'] = le.fit_transform(df['status'])  # SUCCESS=0, FAILURE=1

    # Features
    X = df[['cpu_usage_percent',
            'memory_gb',
            'dependency_delay_min',
            'runtime_seconds',
            'previous_failures']]

    y = df['status']

    # ===============================
    # TRAIN TEST SPLIT
    # ===============================
    print("\nStep 4: Train-Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ===============================
    # SCALING (only for some models)
    # ===============================
    print("\nStep 5: Scaling (for LR, SVM, KNN)")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===============================
    # VIF CHECK
    # ===============================
    print("\nStep 6: VIF Check")

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(X.columns))
    ]

    print(vif_data)

    # ===============================
    # MODEL TRAINING
    # ===============================
    print("\nStep 7: Training Multiple Models")

    results = []

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    for name, model in models.items():

        if name in ["Logistic Regression", "SVM", "KNN"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        results.append((name, acc))

        print(f"{name} Accuracy: {acc}")

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

    print("\nModel Comparison:\n", results_df)

    # ===============================
    # FINAL MODEL (RANDOM FOREST)
    # ===============================
    print("\nStep 8: Selecting Final Model → Random Forest")

    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42
    )

    # Train on original data (NO scaling)
    final_model.fit(X_train, y_train)

    print("Final model training completed ✅")

    # ===============================
    # EVALUATION
    # ===============================
    print("\nStep 9: Evaluation")

    final_pred = final_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, final_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, final_pred))
    print("\nClassification Report:\n", classification_report(y_test, final_pred))

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    print("\nStep 10: Feature Importance")

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": final_model.feature_importances_
    })

    print(feature_importance.sort_values(by="Importance", ascending=False))

    # ===============================
    # SAMPLE PREDICTION
    # ===============================
    print("\nStep 11: Sample Prediction")

    sample = [[88, 4, 12, 200, 5]]

    prediction = final_model.predict(sample)

    print("Prediction (0=SUCCESS, 1=FAILURE):", prediction)

    #===============================
    # SAVE MODEL (MLOPS)
    #===============================
    print("\nStep 12: Saving Model")

    # os.makedirs("models", exist_ok=True)

    # joblib.dump(final_model, "models/autosys_model.pkl")
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent  # go up from src/ to ML folder
    models_dir = BASE_DIR / "models"
    if  models_dir.exists():
        shutil.rmtree( models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_dir / f"autosys_model_{timestamp}.pkl"
    joblib.dump(final_model, model_file)

    print(f"Model saved successfully ✅ at {model_file}")
    if final_model is not None:
        joblib.dump(final_model, models_dir / "autosys_model.pkl")
        print("Model saved successfully ✅")
    else:
        raise ValueError("Training failed: final_model is None")

    print("Model saved successfully ✅")

    # print("\nStep 12: Saving Model")

    # # Set base directory
    # BASE_DIR = Path.cwd()  # safer for GitHub Actions

    # # Models folder
    # models_dir = BASE_DIR / "models"

    # # Remove if exists
    # if models_dir.exists():
    #     shutil.rmtree(models_dir)

    # # Recreate folder
    # models_dir.mkdir(parents=True, exist_ok=True)

    # # Save model with timestamp
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_file = models_dir / f"autosys_model_{timestamp}.pkl"

    # joblib.dump(final_model, model_file)

    # print(f"Model saved successfully ✅ at {model_file}")

    return final_model


# ===============================
# RUN SCRIPT
# ===============================
if __name__ == "__main__":
    train_final_model()