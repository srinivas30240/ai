# AutoSys Job Failure Prediction (Machine Learning Classification)

## 📌 Project Overview

This project focuses on predicting AutoSys batch job failures using Machine Learning.
It helps improve reliability in enterprise batch processing environments by identifying potential failures before execution.

---

## 🎯 Problem Statement

AutoSys jobs may fail due to:

* High CPU usage
* Memory constraints
* Dependency delays
* Long runtime
* Historical job instability

The goal is to build a **classification model** that predicts whether a job will:

* SUCCESS
* FAILURE

---

## 🧠 Approach

* Performed Exploratory Data Analysis (EDA)
* Feature Engineering
* Checked data quality and feature relationships
* Trained multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
* Selected **SVM as final model** based on performance

---

## 📊 Dataset

* 100,000+ AutoSys job records
* Features:

  * cpu_usage_percent
  * memory_gb
  * dependency_delay_min
  * runtime_seconds
  * previous_failures

Target:

```text
status (SUCCESS / FAILURE)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/srinivas30240/ai.git
cd AutoSys_Job_Failure_Prediction

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train Model

```bash
python src/train.py
```

### Predict Job Status

```bash
python src/predict.py
```

---

## 📦 Model Saving (MLOps)

The trained model is saved using joblib:

* autosys_model.pkl
* scaler.pkl

This allows reuse without retraining.

---

## 🔮 Future Improvements

* Integrate ML model with AutoSys pipeline
* Connect with Ansible Automation Platform (AAP)
* Enable automated remediation based on predictions

Proposed flow:

```text
AutoSys → ML Model → Ansible AAP → Auto Remediation
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

---

## 👨‍💻 Author

Srinivas Boga
DevOps | AIOps | Automation | Cloud
