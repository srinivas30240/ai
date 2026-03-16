# AutoSys Job Failure Prediction using Machine Learning

## Project Overview

Enterprise batch environments (banking, telecom, retail, etc.) run **thousands of AutoSys jobs every day**.
Failures or delays in these jobs can cause:

* SLA breaches
* Downstream job failures
* Manual intervention by operations teams
* Incident creation in ITSM systems like ServiceNow

This project builds a **Machine Learning model that predicts AutoSys job failure** using historical execution data.

The goal is to enable **AIOps-driven proactive remediation**, where the system predicts failures before they happen and triggers automation.

---

# Problem Statement

AutoSys jobs may fail or be delayed due to multiple factors such as:

* High CPU usage
* High memory consumption
* Dependency delays
* Long runtime
* Historical instability (previous failures)

The objective is to **train a supervised machine learning model** that predicts whether a job will:

* **SUCCESS**
* **FAILURE**

Using historical AutoSys job execution data.

---

# Why Supervised Learning

The dataset already contains **labeled outputs**.

Example:

| CPU | Memory | Delay | Runtime | Prev Failures | Status  |
| --- | ------ | ----- | ------- | ------------- | ------- |
| 45  | 3      | 0     | 120     | 0             | SUCCESS |
| 88  | 7      | 12    | 300     | 2             | FAILURE |

Because the correct output (**status**) is known, this is a **Supervised Learning classification problem**.

---

# Dataset Description

Dataset size: **120,000 rows**

Columns:

| Column               | Description                      |
| -------------------- | -------------------------------- |
| job_name             | AutoSys job identifier           |
| start_time           | Job execution start time         |
| cpu_usage_percent    | CPU usage during execution       |
| memory_gb            | Memory consumption               |
| dependency_delay_min | Delay due to upstream dependency |
| runtime_seconds      | Job execution time               |
| previous_failures    | Number of past failures          |
| status               | Job result (SUCCESS / FAILURE)   |

Target variable:

```
status
```

---

# Machine Learning Workflow

```
Load Dataset
↓
Exploratory Data Analysis (EDA)
↓
Data Cleaning
↓
Feature Engineering
↓
Train-Test Split
↓
Feature Scaling (for certain models)
↓
Train Multiple Models
↓
Model Comparison
↓
Select Best Model
↓
Final Training
↓
Prediction
```

---

# Step 1: Load Dataset

```python
import pandas as pd

df = pd.read_csv("autosys_dataset_120k.csv")
df.head()
```

---

# Step 2: Exploratory Data Analysis (EDA)

EDA helps understand patterns in the data.

Check dataset structure:

```python
df.shape
df.info()
df.describe()
```

Check missing values:

```python
df.isnull().sum()
```

Check distribution of job results:

```python
df['status'].value_counts()
```

---

# Step 3: Feature Engineering

Convert categorical labels into numeric values.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])

# SUCCESS = 0
# FAILURE = 1
```

---

# Step 4: Feature Selection

```python
X = df[['cpu_usage_percent',
        'memory_gb',
        'dependency_delay_min',
        'runtime_seconds',
        'previous_failures']]

y = df['status']
```

---

# Step 5: Train-Test Split

Split data for training and testing.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Training data: 80%
Testing data: 20%

---

# Step 6: Feature Scaling

Scaling is applied for models that depend on feature magnitude.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# Step 7: Train Multiple Models

Models tested:

* Logistic Regression (Linear)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

Example: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
```

---

# Step 8: Model Evaluation

Use **Accuracy** because this is a **classification problem**.

```python
from sklearn.metrics import accuracy_score

pred = log_model.predict(X_test_scaled)
accuracy_score(y_test, pred)
```

Example model comparison:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 84%      |
| Decision Tree       | 88%      |
| Random Forest       | 92%      |
| SVM                 | 89%      |
| KNN                 | 86%      |

---

# Step 9: Model Selection

Random Forest produced the highest accuracy.

Reasons:

* Handles nonlinear relationships
* Robust to noise
* Handles feature interactions well

Therefore:

```
Random Forest = Final Model
```

---

# Step 10: Final Model Training

```python
from sklearn.ensemble import RandomForestClassifier

final_model = RandomForestClassifier()
final_model.fit(X_train, y_train)
```

---

# Step 11: Predict New Job Outcome

Example new job data:

| CPU | Memory | Delay | Runtime | Prev Fail |
| --- | ------ | ----- | ------- | --------- |
| 88  | 7      | 12    | 300     | 2         |

Prediction:

```python
new_job = [[88,7,12,300,2]]

prediction = final_model.predict(new_job)

print(prediction)
```

Output:

```
1
```

Meaning:

```
FAILURE
```

---

# Feature Importance

Random Forest can show which features influence failures most.

Example:

| Feature              | Importance |
| -------------------- | ---------- |
| dependency_delay_min | High       |
| cpu_usage_percent    | High       |
| previous_failures    | Medium     |
| memory_gb            | Medium     |
| runtime_seconds      | Low        |

Insight:

**Dependency delays and CPU usage are the strongest indicators of job failures.**

---

# Business Impact

Benefits of this system:

* Predict AutoSys job failures before execution
* Reduce SLA breaches
* Enable proactive remediation
* Reduce manual troubleshooting
* Enable **AIOps self-healing systems**

---

# Future Enhancements

Possible improvements:

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Real-time predictions using streaming logs
* Integration with automation tools
* Trigger remediation using Ansible AAP
* Integration with ServiceNow incident automation

---

# Example AIOps Architecture

```
AutoSys Jobs
      ↓
Job Logs / Metrics
      ↓
Data Pipeline
      ↓
Machine Learning Model
      ↓
Failure Prediction
      ↓
Automation (Ansible)
      ↓
ServiceNow Incident / Remediation
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib / Seaborn
* Jupyter Notebook

---

# Author

**Srinivas Boga**
Principal Infrastructure Engineer
DevOps | Automation | Cloud | AIOps
