import streamlit as st
import requests


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="💰",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💰 Medical Insurance Charges Predictor")

st.write(
    "Predict medical insurance charges using "
    "a Decision Tree model deployed on Azure ML."
)


# --------------------------------------------------
# AZURE CONFIGURATION
# --------------------------------------------------

endpoint_url = st.secrets["AZURE_ENDPOINT"]

api_key = st.secrets["AZURE_API_KEY"]


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

st.subheader("Customer Information")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)


sex = st.selectbox(
    "Sex",
    [
        "male",
        "female"
    ]
)


bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)


children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)


smoker = st.selectbox(
    "Smoker",
    [
        "yes",
        "no"
    ]
)


claim_amount = st.number_input(
    "Claim Amount",
    min_value=0.0,
    value=0.0
)


past_consultations = st.number_input(
    "Past Consultations",
    min_value=0,
    value=0
)


num_of_steps = st.number_input(
    "Number of Steps",
    min_value=0,
    value=5000
)


hospital_expenditure = st.number_input(
    "Hospital Expenditure",
    min_value=0.0,
    value=0.0
)


past_hospitalizations = st.number_input(
    "Number of Past Hospitalizations",
    min_value=0,
    value=0
)


annual_salary = st.number_input(
    "Annual Salary",
    min_value=0.0,
    value=600000.0
)


region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Insurance Charges",
    type="primary"
):

    payload = {

        "data": [

            {
                "age": age,

                "sex": sex,

                "bmi": bmi,

                "children": children,

                "smoker": smoker,

                "Claim_Amount": claim_amount,

                "past_consultations":
                    past_consultations,

                "num_of_steps":
                    num_of_steps,

                "Hospital_expenditure":
                    hospital_expenditure,

                "NUmber_of_past_hospitalizations":
                    past_hospitalizations,

                "Anual_Salary":
                    annual_salary,

                "region": region
            }

        ]

    }


    # Azure authentication
    headers = {

        "Content-Type":
            "application/json",

        "Authorization":
            f"Bearer {api_key}"

    }


    try:

        response = requests.post(

            endpoint_url,

            json=payload,

            headers=headers,

            timeout=30

        )


        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "predictions"
            ][0]


            st.success(
                "Prediction successful!"
            )


            st.metric(
                "Predicted Insurance Charges",
                f"₹ {prediction:,.2f}"
            )


        else:

            st.error(
                f"Azure Error "
                f"{response.status_code}"
            )

            st.code(
                response.text
            )


    except requests.exceptions.RequestException as e:

        st.error(
            "Unable to connect to Azure endpoint."
        )

        st.exception(e)
