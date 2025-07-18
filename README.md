# Inventory Risk Prediction using ANN

A machine learning project to classify inventory as "stockout" or "overstock" using an Artificial Neural Network. Trained using SMOTE-balanced data and deployed with a Streamlit web interface.

## Features Used
- current_stock
- reorder_point
- avg_daily_demand
- sales_last_30_days
- stock_turnover_ratio
- lead_time_days
- forecast_error

## Tools Used
- Python, TensorFlow, scikit-learn, imbalanced-learn
- Streamlit for UI

## Run the App
```bash
streamlit run streamlit_app/app.py
