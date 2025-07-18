import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("inventory_ann_top7_model.keras")
scaler = joblib.load("scaler_top7.pkl")

st.title("📦 Inventory Risk Predictor (Top 7 Features)")

cols = [
    'current_stock', 'reorder_point', 'avg_daily_demand',
    'sales_last_30_days', 'stock_turnover_ratio',
    'lead_time_days', 'forecast_error'
]

user_input = []
for col in cols:
    val = st.number_input(f"{col.replace('_', ' ').title()}", step=1.0 if 'ratio' in col or 'error' in col else 1)
    user_input.append(val)

if st.button("Predict Risk"):
    df_input = pd.DataFrame([user_input], columns=cols)
    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)[0][0]
    label = "Stockout" if pred >= 0.5 else "Overstock"
    st.success(f"Predicted: **{label}**")
    st.info(f"Probability of Stockout: **{pred:.2f}**")
