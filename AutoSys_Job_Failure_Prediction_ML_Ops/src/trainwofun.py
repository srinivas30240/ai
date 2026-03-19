# Data processing
import pandas as pd
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score
# save the model
import joblib

print("All Pacakge are installed succesfully")

#Load the dataset
df = pd.read_csv("C:/Users/Shrinivas/Desktop/training/code_python/vectordatabase/AutoSys_Job_Failure_Prediction/data/autosys_dataset.csv")

#Display top 5 
print(df.head())
#print(df)

#Understand Dataset Structure , check the nummber of rows and columns
print(f"Number of rows and columns: {df.shape}")
# check the columns
print(df.columns)
# Check data types
print(df.dtypes)
# Check Missing Values &  Count missing values
print(df.isnull().sum())
#Check Target Variable Distribution & Count success vs failure
print(df['status'].value_counts())

#Step 7 — Feature Engineering: Convert categorical column status → numeric. Convert SUCCESS / FAILURE to numbers Success = 0, Failure = 1
le = LabelEncoder()
df["status"] = le.fit_transform(df['status'])
print(df['status'])
#feature columns
X =df[['cpu_usage_percent',
        'memory_gb',
        'dependency_delay_min',
        'runtime_seconds',
        'previous_failures']]
y = df['status'] #target value
# train/split test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled, X_test_scaled)
print(y)

print("Data has been standardized.")
#Calculate VIF
# Create empty dataframe for results
vif_data = pd.DataFrame()

# Store feature names
vif_data["Feature"] = X.columns

# Calculate VIF for each feature
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(len(X.columns))
]

print(vif_data)
#Logistic Regression (Linear)
log_model = LogisticRegression()

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, log_pred)
# Decision Tree
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
#random forest
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
#SVM
svm_model = SVC()

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
#KNN
knn_model = KNeighborsClassifier()

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_acc = accuracy_score(y_test, knn_pred)
#Compare All Models
results = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    "Accuracy":[
        log_acc,
        dt_acc,
        rf_acc,
        svm_acc,
        knn_acc
    ]
})

print(results)
#11:  highest accuracy, so SVM is currently the best model among these.
# Final Model (based on best accuracy)
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# Train on full training data
final_model.fit(X_train, y_train)
print("Final model training completed")
#make predections
final_pred = final_model.predict(X_test)
print("predication:", final_pred)

# final accuracy of model 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, final_pred)
print("Accuracy:", accuracy)
# Predictions
y_pred = final_model.predict(X_test)

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Precision, Recall, F1-score
print(classification_report(y_test, y_pred))

# Predict New AutoSys Job
new_job = [[88, 4,12, 200, 5]]
predication =  final_model.predict(new_job)
print(predication)
sample = pd.DataFrame([[85,4,10,300,2]], columns=X.columns)

dp=final_model.predict(sample)
print("data_prediction", dp )

#SAVE MODEL + SCALER
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/autosys_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully ✅")