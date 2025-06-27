
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Simulate monthly sales data
def generate_data():
    months = pd.date_range(start='2023-01-01', periods=30, freq='MS')
    sales = [450 + i*5 + np.random.randint(-10, 10) for i in range(30)]
    df = pd.DataFrame({'Month': months, 'Sales_kg': sales})
    return df

# Forecast future sales
def forecast_sales(df, months_ahead):
    df = df.copy()
    df['Month_Index'] = np.arange(len(df))
    X = df[['Month_Index']]
    y = df['Sales_kg']

    model = LinearRegression()
    model.fit(X, y)
    future_indexes = np.arange(len(df), len(df) + months_ahead).reshape(-1, 1)
    future_sales = model.predict(future_indexes)
    future_months = pd.date_range(start=df['Month'].iloc[-1] + pd.offsets.MonthBegin(), periods=months_ahead, freq='MS')
    future_df = pd.DataFrame({'Month': future_months, 'Predicted_Sales_kg': future_sales})

    r2 = r2_score(y, model.predict(X))
    return df, future_df, r2

# Streamlit App UI
st.title("📈 Fish Sales Forecasting - Madeco Company")

df = generate_data()
st.subheader("Historical Sales Data")
st.line_chart(df.set_index('Month'))

months_to_forecast = st.slider("Months to forecast:", 1, 12, 6)

df_actual, df_future, r2 = forecast_sales(df, months_to_forecast)

# Plotting combined actual + forecast
combined_df = pd.concat([
    df_actual[['Month', 'Sales_kg']].rename(columns={'Sales_kg': 'kg'}).assign(Type='Actual'),
    df_future.rename(columns={'Predicted_Sales_kg': 'kg'}).assign(Type='Forecast')
])

pivot_df = combined_df.pivot(index='Month', columns='Type', values='kg')
st.subheader("Forecasted Sales")
st.line_chart(pivot_df)

st.markdown(f"**Model R² Score:** `{r2:.4f}`")

st.dataframe(df_future.set_index('Month').round(2))
