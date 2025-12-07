import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained wellness-tourism model
model_path = hf_hub_download(
    repo_id="vsakar/wellness_tourism_model",
    filename="prod_wellness_model.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction")
st.write("""
This application predicts whether a customer is likely to **purchase a wellness tourism package**
based on their demographic and behavioral characteristics.
Fill in the details below to get a prediction.
""")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
num_trips = st.number_input("Number of Trips Taken", min_value=0, max_value=50, value=2)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000, step=500)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=10, value=7)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=3)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=30)

typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Self_Employed", "Student", "Housewife"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Designation": designation,
    "ProductPitched": product_pitched
}])

# Predict button
if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("✅ Customer is likely to purchase the wellness tourism package!")
    else:
        st.warning("❌ Customer is unlikely to purchase the package.")
