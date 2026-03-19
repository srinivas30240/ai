# ===============================
# IMPORT
# ===============================
import joblib

# ===============================
# LOAD MODEL
# ===============================
def load_model():
    return joblib.load("models/autosys_model.pkl")


# ===============================
# PREDICT FUNCTION
# ===============================
def predict_job(model, data):
    return "FAILURE" if model.predict([data])[0] == 1 else "SUCCESS"


# ===============================
# MAIN
# ===============================
def main():

    print("Loading trained model...")
    model = load_model()

    sample = [89, 11, 12, 600, 6]

    result = predict_job(model, sample)

    print("Prediction:", result)


if __name__ == "__main__":
    main()