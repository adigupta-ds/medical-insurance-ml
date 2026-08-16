import joblib
import pandas as pd


MODEL_PATH = "model/model.pkl"


# Load model
model = joblib.load(MODEL_PATH)


# Example customer
data = {
    "age": [30],
    "sex": ["male"],
    "bmi": [25.5],
    "children": [1],
    "smoker": ["no"],
    "Claim_Amount": [5000],
    "past_consultations": [2],
    "num_of_steps": [5000],
    "Hospital_expenditure": [10000],
    "NUmber_of_past_hospitalizations": [1],
    "Anual_Salary": [600000],
    "region": ["southwest"]
}


df = pd.DataFrame(data)


# Prediction
prediction = model.predict(df)


print("Predicted Insurance Charges:")
print(prediction[0])
