import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="AI Fraud Detection",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #1fffdd, #ffd6b3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(" AI Fraud Detection Dashboard")

st.markdown("---")

# SIDEBAR
st.sidebar.header("Transaction Inputs")

features = []

for i in range(30):
    value = st.sidebar.number_input(
        f"Feature {i}",
        value=0.0
    )
    features.append(value)

# MAIN BUTTON
if st.sidebar.button("Analyze Transaction"):

    payload = {
        "features": features
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        prediction = result["fraud_prediction"]

        probability = result["fraud_probability"]

        # RESULT SECTION
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("Fraudulent Transaction")
            else:
                st.success("Legitimate Transaction")

            st.metric(
                label="Fraud Probability",
                value=f"{probability:.2%}"
            )

        # GAUGE CHART
        with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Fraud Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'thickness': 0.3},
                    'steps': [
                        {'range': [0, 30]},
                        {'range': [30, 70]},
                        {'range': [70, 100]}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # FEATURE VISUALIZATION
        st.subheader("Feature Distribution")

        df = pd.DataFrame({
            "Feature": [f"F{i}" for i in range(30)],
            "Value": features
        })

        fig2 = px.bar(
            df,
            x="Feature",
            y="Value",
            title="Input Feature Values"
        )

        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error connecting to backend: {e}")