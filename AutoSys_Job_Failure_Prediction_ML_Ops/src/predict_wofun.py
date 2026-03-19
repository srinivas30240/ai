# ===============================
# STEP 1: IMPORT
# ===============================
import joblib

# ===============================
# STEP 2: LOAD MODEL
# ===============================
model = joblib.load("models/autosys_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===============================
# STEP 3: PREDICTION FUNCTION
# ===============================
def predict_job(data):
    data_scaled = scaler.transform([data])
    result = model.predict(data_scaled)

    return "FAILURE" if result[0] == 1 else "SUCCESS"

# ===============================
# STEP 4: TEST PREDICTION
# ===============================
sample = [88,7,12,300,2]

print("Prediction:", predict_job(sample))