import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ice Cream Revenue Predictor", page_icon="🍦")

# ---- CUSTOM DARK AI THEME CSS ----
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    h1 {
        color: #00f2ff;
        text-align: center;
    }

    .stButton>button {
        background-color: #00f2ff;
        color: black;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        font-size: 16px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #00c3cc;
        box-shadow: 0px 0px 15px #00f2ff;
    }

    .prediction-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #00f2ff;
        margin-top: 20px;
        box-shadow: 0px 0px 20px rgba(0,242,255,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL ----
model = joblib.load("clf.pkl")

# ---- HEADER ----
st.markdown("<h1>🍦 Ice Cream Revenue Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered revenue prediction based on temperature</p>", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.header("📌 Project Info")
st.sidebar.write("Model: Linear Regression")
st.sidebar.write("Theme: Dark AI")
st.sidebar.write("Built by Kamran Hyder")

# ---- INPUT ----
st.subheader("Enter Temperature")

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    format="%.2f"
)

# ---- PREDICTION ----
if st.button("Predict Revenue 🚀"):

    input_data = np.array([[temperature]])
    prediction = model.predict(input_data)[0]

    # Styled prediction box
    st.markdown(
        f"<div class='prediction-box'>💰 Predicted Revenue: ${prediction:.2f}</div>",
        unsafe_allow_html=True
    )

    # ---- GRAPH ----
    temps = np.linspace(0, 50, 100)
    revenues = model.predict(temps.reshape(-1, 1))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.plot(temps, revenues)
    ax.scatter(temperature, prediction)

    ax.set_xlabel("Temperature (°C)", color="white")
    ax.set_ylabel("Revenue", color="white")
    ax.set_title("Revenue vs Temperature", color="#00f2ff")

    ax.tick_params(colors='white')

    st.pyplot(fig)