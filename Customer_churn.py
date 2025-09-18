
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    with open('churn_prediction_model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

best_model = load_model()

st.title('Customer Churn Predictor 🔮')
st.write("Enter the customer's details below to predict whether they will churn or not.")
st.write("This app uses a pre-trained XGBoost model to make predictions.")
st.sidebar.header('Customer Details')

def user_input_features():
    account_length = st.sidebar.slider('Account Length (days)', 1, 240, 100)
    customer_service_calls = st.sidebar.slider('Customer Service Calls', 0, 10, 1)
    voice_mail_plan = st.sidebar.selectbox('Voice Mail Plan', ('No', 'Yes'))
    international_plan = st.sidebar.selectbox('International Plan', ('No', 'Yes'))
    voice_mail_messages = st.sidebar.number_input('Number of Voice Mail Messages', 0, 60, 0)
    day_mins = st.sidebar.number_input('Total Day Minutes', 0.0, 400.0, 180.0)
    evening_mins = st.sidebar.number_input('Total Evening Minutes', 0.0, 400.0, 200.0)
    night_mins = st.sidebar.number_input('Total Night Minutes', 0.0, 400.0, 200.0)
    international_mins = st.sidebar.number_input('Total International Minutes', 0.0, 25.0, 10.0)
    day_calls = st.sidebar.number_input('Total Day Calls', 0, 200, 100)
    day_charge = st.sidebar.number_input('Total Day Charge ($)', 0.0, 70.0, 30.0)
    evening_calls = st.sidebar.number_input('Total Evening Calls', 0, 200, 100)
    evening_charge = st.sidebar.number_input('Total Evening Charge ($)', 0.0, 40.0, 17.0)
    night_calls = st.sidebar.number_input('Total Night Calls', 0, 200, 100)
    night_charge = st.sidebar.number_input('Total Night Charge ($)', 0.0, 20.0, 9.0)
    international_calls = st.sidebar.number_input('Total International Calls', 0, 20, 4)
    international_charge = st.sidebar.number_input('Total International Charge ($)', 0.0, 6.0, 2.7)
    total_charge = day_charge + evening_charge + night_charge + international_charge
    total_mins = day_mins + evening_mins + night_mins + international_mins
    total_calls = day_calls + evening_calls + night_calls + international_calls
    mins_per_call = total_mins / total_calls if total_calls > 0 else 0

    data = {
        'account_length': account_length,
        'voice_mail_plan': 1 if voice_mail_plan == 'Yes' else 0,
        'voice_mail_messages': voice_mail_messages,
        'day_mins': day_mins,
        'evening_mins': evening_mins,
        'night_mins': night_mins,
        'international_mins': international_mins,
        'customer_service_calls': customer_service_calls,
        'international_plan': 1 if international_plan == 'Yes' else 0,
        'day_calls': day_calls,
        'day_charge': day_charge,
        'evening_calls': evening_calls,
        'evening_charge': evening_charge,
        'night_calls': night_calls,
        'night_charge': night_charge,
        'international_calls': international_calls,
        'international_charge': international_charge,
        'total_charge': total_charge,
        'total_mins': total_mins,
        'total_calls': total_calls,
        'mins_per_call': mins_per_call
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input')
st.write(input_df)

# Reorder columns to exactly match model's expected features
expected_features = best_model.get_booster().feature_names
input_df = input_df[expected_features]

if st.button('Predict Churn'):
    prediction = best_model.predict(input_df)
    prediction_proba = best_model.predict_proba(input_df)
    if prediction[0] == 1:
        st.error(f'Prediction: **This customer is likely to CHURN.**')
        st.write(f"Confidence Score: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.success(f'Prediction: **This customer is likely to STAY.**')
        st.write(f"Confidence Score: **{prediction_proba[0][0]*100:.2f}%**")
