import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")

st.markdown("""
### 📊 Model Information

**Model Used:** Gradient Boosting Classifier

**Purpose:** Predict whether a bank customer is likely to churn based on customer demographics, account information, and banking behavior.

---
""")
st.markdown("---")

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------

st.sidebar.header("Customer Information")

credit_score = st.sidebar.slider("Credit Score",300,900,650)

age = st.sidebar.slider("Age",18,100,40)

tenure = st.sidebar.slider("Tenure (Years)",0,10,5)

balance = st.sidebar.number_input("Balance",0.0,300000.0,50000.0)

products = st.sidebar.selectbox(
    "Number of Products",
    [1,2,3,4]
)

credit_card = st.sidebar.selectbox(
    "Has Credit Card?",
    [0,1]
)

active_member = st.sidebar.selectbox(
    "Active Member?",
    [0,1]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    0.0,
    300000.0,
    100000.0
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female","Male"]
)

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

balance_salary_ratio = balance/(salary+1)

has_multiple_products = 1 if products>1 else 0

tenure_product_interaction = tenure*products

age_tenure_interaction = age*tenure

geo_germany = 1 if geography=="Germany" else 0

geo_spain = 1 if geography=="Spain" else 0

gender_male = 1 if gender=="Male" else 0

# ---------------------------------------------------
# Create DataFrame
# ---------------------------------------------------

input_df = pd.DataFrame({

'Year':[2025],

'CreditScore':[credit_score],

'Age':[age],

'Tenure':[tenure],

'Balance':[balance],

'NumOfProducts':[products],

'HasCrCard':[credit_card],

'IsActiveMember':[active_member],

'EstimatedSalary':[salary],

'Geography_Germany':[geo_germany],

'Geography_Spain':[geo_spain],

'Gender_Male':[gender_male],

'BalanceToSalaryRatio':[balance_salary_ratio],

'HasMultipleProducts':[has_multiple_products],

'TenureProductInteraction':[tenure_product_interaction],

'AgeTenureInteraction':[age_tenure_interaction]

})

# ---------------------------------------------------
# Scale Features
# ---------------------------------------------------

scaled_input = scaler.transform(input_df)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
st.subheader("📋 Customer Summary")

summary = {
    "Credit Score": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": f"${balance:,.2f}",
    "Products": products,
    "Credit Card": "Yes" if credit_card else "No",
    "Active Member": "Yes" if active_member else "No",
    "Estimated Salary": f"${salary:,.2f}",
    "Geography": geography,
    "Gender": gender
}

st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))

if st.button("Predict Churn"):

    probability = model.predict_proba(scaled_input)[0][1]
    prediction = model.predict(scaled_input)[0]

    st.subheader("📈 Prediction Results")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if probability >= 0.70:
        st.error("🔴 HIGH RISK CUSTOMER")

        st.markdown("""
### 💡 Recommended Action

- Contact the customer immediately.
- Offer loyalty rewards or discounts.
- Assign a relationship manager.
- Monitor account activity closely.
""")

    elif probability >= 0.40:
        st.warning("🟡 MEDIUM RISK CUSTOMER")

        st.markdown("""
### 💡 Recommended Action

- Send personalized offers.
- Improve customer engagement.
- Monitor future transactions.
""")

    else:
        st.success("🟢 LOW RISK CUSTOMER")

        st.markdown("""
### 💡 Recommended Action

- Customer is likely to stay.
- Continue providing quality service.
- Offer premium banking products.
""")

    st.progress(float(probability))

st.markdown("---")

st.caption("""
Developed by **Sharmila KS**

Unified Mentor Internship Project

Domain: Data Analytics

""")

