import os
import json
import joblib
import pandas as pd


model = None


# --------------------------------------------------
# INITIALIZE MODEL
# --------------------------------------------------

def init():

    global model

    model_path = os.path.join(
        os.environ["AZUREML_MODEL_DIR"],
        "model.pkl"
    )

    model = joblib.load(model_path)

    print("Model loaded successfully")


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def run(raw_data):

    try:

        # Convert incoming JSON string
        if isinstance(raw_data, str):
            data = json.loads(raw_data)

        else:
            data = raw_data


        # Expected:
        #
        # {
        #     "data": [
        #         {
        #             "age": 30,
        #             ...
        #         }
        #     ]
        # }


        input_data = data["data"]


        # Convert to DataFrame
        df = pd.DataFrame(input_data)


        # Prediction
        predictions = model.predict(df)


        # Return JSON
        return {
            "predictions": predictions.tolist()
        }


    except Exception as e:

        return {
            "error": str(e)
        }
