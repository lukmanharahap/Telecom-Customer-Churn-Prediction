import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import random

model = joblib.load('model/random_forest_churn_model.pkl')
feature_importance = model.feature_importances_

feature_names = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure",
    "Phone Service", "Multiple Lines", "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Paperless Billing", "Monthly Charges", "Total Charges",
    "Internet Service: DSL", "Internet Service: Fiber Optic", "Internet Service: No",
    "Contract: Month-to-Month", "Contract: One Year", "Contract: Two Year",
    "Payment Method: Bank Transfer", "Payment Method: Credit Card",
    "Payment Method: Electronic Check", "Payment Method: Mailed Check"
]

st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

st.sidebar.header("Bulk Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.write("### Download Sample CSV")
sample_data = []
for _ in range(50):
    internet_service = random.choice(['DSL', 'Fiber', 'No'])
    contract = random.choice(['Month', 'OneYear', 'TwoYear'])
    payment_method = random.choice(['Bank', 'Credit', 'Electronic', 'Mailed'])

    entry = {
    "gender": random.choice(["Female", "Male"]),
    "SeniorCitizen": random.choice(["Yes", "No"]),
    "Partner": random.choice(["Yes", "No"]),
    "Dependents": random.choice(["Yes", "No"]),
    "tenure": random.randint(0, 72),
    "PhoneService": random.choice(["Yes", "No"]),
    "MultipleLines": random.choice(["Yes", "No"]),
    "OnlineSecurity": random.choice(["Yes", "No"]),
    "OnlineBackup": random.choice(["Yes", "No"]),
    "DeviceProtection": random.choice(["Yes", "No"]),
    "TechSupport": random.choice(["Yes", "No"]),
    "StreamingTV": random.choice(["Yes", "No"]),
    "StreamingMovies": random.choice(["Yes", "No"]),
    "PaperlessBilling": random.choice(["Yes", "No"]),
    "MonthlyCharges": round(random.uniform(0.0, 120.0), 2),
    "TotalCharges": round(random.uniform(0.0, 8700.0), 2),
    "InternetService_DSL": (internet_service == 'DSL'),
    "InternetService_Fiber optic": (internet_service == 'Fiber'),
    "InternetService_No": (internet_service == 'No'),
    "Contract_Month-to-month": (contract == 'Month'),
    "Contract_One year": (contract == 'OneYear'),
    "Contract_Two year": (contract == 'TwoYear'),
    "PaymentMethod_Bank transfer (automatic)": (payment_method == 'Bank'),
    "PaymentMethod_Credit card (automatic)": (payment_method == 'Credit'),
    "PaymentMethod_Electronic check": (payment_method == 'Electronic'),
    "PaymentMethod_Mailed check": (payment_method == 'Mailed'),
    }
    sample_data.append(entry)

sample_df = pd.DataFrame(sample_data)

st.sidebar.download_button(
    label="Download Sample CSV",
    data=sample_df.to_csv(index=False, header=True),
    file_name="sample.csv",
    mime="text/csv",
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['gender'] = df['gender'].map({"Male": 1, "Female": 0})
    df['SeniorCitizen'] = df['SeniorCitizen'].map({"Yes": 1, "No": 0})
    df['Partner'] = df['Partner'].map({"Yes": 1, "No": 0})
    df['Dependents'] = df['Dependents'].map({"Yes": 1, "No": 0})
    df['PhoneService'] = df['PhoneService'].map({"Yes": 1, "No": 0})
    df['MultipleLines'] = df['MultipleLines'].map({"Yes": 1, "No": 0})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({"Yes": 1, "No": 0})
    df['OnlineBackup'] = df['OnlineBackup'].map({"Yes": 1, "No": 0})
    df['DeviceProtection'] = df['DeviceProtection'].map({"Yes": 1, "No": 0})
    df['TechSupport'] = df['TechSupport'].map({"Yes": 1, "No": 0})
    df['StreamingTV'] = df['StreamingTV'].map({"Yes": 1, "No": 0})
    df['StreamingMovies'] = df['StreamingMovies'].map({"Yes": 1, "No": 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({"Yes": 1, "No": 0})
    df['InternetService_DSL'] = df['InternetService_DSL'].map({True: 1, False: 0})
    df['InternetService_Fiber optic'] = df['InternetService_Fiber optic'].map({True: 1, False: 0})
    df['InternetService_No'] = df['InternetService_No'].map({True: 1, False: 0})
    df['Contract_Month-to-month'] = df['Contract_Month-to-month'].map({True: 1, False: 0})
    df['Contract_One year'] = df['Contract_One year'].map({True: 1, False: 0})
    df['Contract_Two year'] = df['Contract_Two year'].map({True: 1, False: 0})
    df['PaymentMethod_Bank transfer (automatic)'] = df['PaymentMethod_Bank transfer (automatic)'].map({True: 1, False: 0})
    df['PaymentMethod_Credit card (automatic)'] = df['PaymentMethod_Credit card (automatic)'].map({True: 1, False: 0})
    df['PaymentMethod_Electronic check'] = df['PaymentMethod_Electronic check'].map({True: 1, False: 0})
    df['PaymentMethod_Mailed check'] = df['PaymentMethod_Mailed check'].map({True: 1, False: 0})

    predictions = model.predict(df)
    predictions_proba = model.predict_proba(df)
    df["Churn Prediction"] = [p for p in predictions]
    df["Churn Probability"] = predictions_proba[:, 1]
    st.write("### Predictions for Uploaded Data:")
    st.dataframe(df[["Churn Prediction", "Churn Probability"]])

with st.form("churn_form"):
    st.subheader("Enter Customer Details")

    with st.expander("Personal Details"):
        gender = st.radio("Gender:", ["Male", "Female"], horizontal=True)
        gender = 1 if gender == "Male" else 0
        SeniorCitizen = st.radio("Senior Citizen?", ["Yes", "No"], horizontal=True)
        SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
        Partner = st.radio("Having a Partner?", ["Yes", "No"], horizontal=True)
        Partner = 1 if Partner == "Yes" else 0
        Dependents = st.radio("Having Dependents?", ["Yes", "No"], horizontal=True)
        Dependents = 1 if Dependents == "Yes" else 0

    with st.expander("Services"):
        PhoneService = st.radio("Have Phone Service?", ["Yes", "No"], horizontal=True)
        PhoneService = 1 if PhoneService == "Yes" else 0
        MultipleLines = st.radio("Multiple Lines?", ["Yes", "No"], horizontal=True)
        MultipleLines = 1 if MultipleLines == "Yes" else 0
        OnlineSecurity = st.radio("Online Security?", ["Yes", "No"], horizontal=True)
        OnlineSecurity = 1 if OnlineSecurity == "Yes" else 0
        OnlineBackup = st.radio("Online Backup?", ["Yes", "No"], horizontal=True)
        OnlineBackup = 1 if OnlineBackup == "Yes" else 0
        DeviceProtection = st.radio("Device Protection?", ["Yes", "No"], horizontal=True)
        DeviceProtection = 1 if DeviceProtection == "Yes" else 0
        TechSupport = st.radio("Tech Support?", ["Yes", "No"], horizontal=True)
        TechSupport = 1 if TechSupport == "Yes" else 0
        StreamingTV = st.radio("Streaming TV?", ["Yes", "No"], horizontal=True)
        StreamingTV = 1 if StreamingTV == "Yes" else 0
        StreamingMovies = st.radio("Streaming Movies?", ["Yes", "No"], horizontal=True)
        StreamingMovies = 1 if StreamingMovies == "Yes" else 0

    with st.expander("Billing Information"):
        PaperlessBilling = st.radio("Paperless Billing?", ["Yes", "No"], horizontal=True)
        PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0
        internetService = st.radio("Internet Service?", ["No", "DSL", "Fiber Optic"], horizontal=True)
        InternetService_DSL = 1 if internetService == "DSL" else 0
        InternetService_Fiber = 1 if internetService == "Fiber Optic" else 0
        InternetService_No = 1 if internetService == "No" else 0
        contract = st.radio("Contract:", ["Month-to-month", "One year", "Two year"], horizontal=True)
        Contract_Month = 1 if contract == "Month-to-month" else 0
        Contract_OneYear = 1 if contract == "One year" else 0
        Contract_TwoYear = 1 if contract == "Two year" else 0
        paymentMethod = st.radio(
            "Payment Method:",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card"],
            horizontal=True
        )
        PaymentMethod_Electronic = 1 if paymentMethod == "Electronic check" else 0
        PaymentMethod_Mailed = 1 if paymentMethod == "Mailed check" else 0
        PaymentMethod_Bank = 1 if paymentMethod == "Bank transfer (automatic)" else 0
        PaymentMethod_Credit = 1 if paymentMethod == "Credit card" else 0
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=120.0, value=50.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=8700.0, value=1000.0)

    submitted = st.form_submit_button("Predict")


if submitted:
    try:
        features = [
            gender,
            SeniorCitizen,
            Partner,
            Dependents,
            tenure,
            PhoneService,
            MultipleLines,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            PaperlessBilling,
            MonthlyCharges,
            TotalCharges,
            InternetService_DSL,
            InternetService_Fiber,
            InternetService_No,
            Contract_Month,
            Contract_OneYear,
            Contract_TwoYear,
            PaymentMethod_Bank,
            PaymentMethod_Credit,
            PaymentMethod_Electronic,
            PaymentMethod_Mailed,
        ]
        with st.spinner("Predicting..."):
            prediction = model.predict([features])
            prediction_proba = model.predict_proba([features])[0][1]
    except Exception as e:
        st.error(f"An error occurred: {e}")


    st.subheader("Prediction Result:")
    if prediction == "Yes":
        st.error("⚠️ **Customer WILL churn**")
    else:
        st.success("✅ **Customer will NOT churn**")

    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    st.write("### Feature Contribution (SHAP Explanation)")
    features_df = pd.DataFrame([features], columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap_values_explanation = shap.Explanation(
        values=shap_values[0][:, 0],
        base_values=explainer.expected_value[0],
        feature_names=features_df.columns
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap_values_explanation)
    st.pyplot(fig)

st.markdown("---")
st.write("Made with ❤️ and Streamlit by Lukman Harahap")