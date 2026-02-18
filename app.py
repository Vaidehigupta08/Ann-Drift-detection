import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Drift Detection", layout="centered")

st.title("📊 Data Drift Detection")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head())

    # numeric only
    df = df.select_dtypes(include=[np.number])

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    # sampling (IMPORTANT)
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)

    payload = {
        "columns": df.columns.tolist(),
        "data": df.values.tolist()
    }

    response = requests.post(
        "http://localhost:8000/detect_drift",
        json=payload,
        timeout=120
    )

    st.write("Status:", response.status_code)

    if response.status_code == 200:
        res = response.json()
        st.write("Drift Ratio:", res["drift_ratio"])

        if res["drift_detected"]:
            st.error("🚨 Drift Detected")
        else:
            st.success("✅ No Drift")
    else:
        st.error(response.text)
