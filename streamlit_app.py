import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import timedelta

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA


# ---------- Title ----------
st.title("Sales & Demand Forecasting for Medicines")


# ---------- Load Data ----------
possible_paths = ["sample_medicine_sales.csv", "sample_medical_data.csv", "multi_medicine_forecast.csv"]
loaded = False
for p in possible_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        loaded = True
        st.sidebar.write(f"Loaded dataset: {p}")
        break

if not loaded:
    st.error("No data file found. Ensure one of sample_medicine_sales.csv, sample_medical_data.csv, or multi_medicine_forecast.csv is present.")
    st.stop()


df["Date"] = pd.to_datetime(df["Date"])


# ---------- Sidebar ----------
st.sidebar.header("Forecast Settings")

medicines = df["Medicine"].unique()

medicine = st.sidebar.selectbox(
    "Select Medicine",
    medicines
)

model_name = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost","Linear Regression","ARIMA"]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    7,
    30,
    14
)


# ---------- Run Forecast ----------
if st.button("Run Forecast"):

    # ---------- Single Medicine Forecast ----------
    df_med = df[df["Medicine"] == medicine].sort_values("Date")
    df_med["DayOfWeek"] = df_med["Date"].dt.dayofweek

    X = df_med[["Campaign","Holiday","DayOfWeek"]]
    y = df_med["Units_Sold"]

    if model_name == "XGBoost":

        model = XGBRegressor(objective="reg:squarederror")
        model.fit(X,y)

        future_dates = [
            df_med["Date"].max() + timedelta(days=i)
            for i in range(1, forecast_days+1)
        ]

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Campaign": np.random.choice([0,1], forecast_days),
            "Holiday": np.random.choice([0,1], forecast_days)
        })

        future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek

        forecast_units = model.predict(
            future_df[["Campaign","Holiday","DayOfWeek"]]
        )

    elif model_name == "Linear Regression":

        lr = LinearRegression()
        lr.fit(np.arange(len(y)).reshape(-1,1), y)

        forecast_units = lr.predict(
            np.arange(len(y), len(y)+forecast_days).reshape(-1,1)
        )

        future_dates = [
            df_med["Date"].max() + timedelta(days=i)
            for i in range(1, forecast_days+1)
        ]

    elif model_name == "ARIMA":

        model = ARIMA(y, order=(5,1,0))
        res = model.fit()

        forecast_units = res.forecast(steps=forecast_days)

        future_dates = [
            df_med["Date"].max() + timedelta(days=i)
            for i in range(1, forecast_days+1)
        ]


    # ---------- Forecast DataFrame ----------
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Units": forecast_units
    })

    unit_price = df_med["Unit_Price"].iloc[-1]

    forecast_df["Revenue"] = forecast_df["Forecast_Units"] * unit_price
    forecast_df["Medicine"] = medicine


    # ---------- Forecast Table ----------
    st.subheader("Forecast Table")
    st.dataframe(forecast_df)


    # ---------- Forecast Graph ----------
    st.subheader("Demand Forecast Graph")

    plt.figure(figsize=(10,5))

    plt.plot(df_med["Date"], df_med["Units_Sold"], label="Historical")
    plt.plot(forecast_df["Date"], forecast_df["Forecast_Units"], label="Forecast")

    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.title("Medicine Demand Forecast")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)


    # ---------- Business Observations ----------
    st.subheader("Business Observations")

    avg_units = forecast_df["Forecast_Units"].mean()
    max_units = forecast_df["Forecast_Units"].max()

    max_day = forecast_df.loc[
        forecast_df["Forecast_Units"].idxmax(),"Date"
    ]

    avg_revenue = forecast_df["Revenue"].mean()
    max_revenue = forecast_df["Revenue"].max()

    st.write(f"Average forecasted units per day: **{int(avg_units)}**")
    st.write(f"Peak demand expected on **{max_day.date()}** with **{int(max_units)} units**")
    st.write(f"Average daily revenue: **₹{avg_revenue:.2f}**")
    st.write(f"Maximum forecast revenue: **₹{max_revenue:.2f}**")


    # ---------- Multi Medicine Forecast ----------
    all_forecasts = []

    for med in medicines:

        df_med = df[df["Medicine"] == med].sort_values("Date")
        df_med["DayOfWeek"] = df_med["Date"].dt.dayofweek

        X = df_med[["Campaign","Holiday","DayOfWeek"]]
        y = df_med["Units_Sold"]

        model = XGBRegressor(objective="reg:squarederror")
        model.fit(X,y)

        future_dates = [
            df_med["Date"].max() + timedelta(days=i)
            for i in range(1, forecast_days+1)
        ]

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Campaign": np.random.choice([0,1], forecast_days),
            "Holiday": np.random.choice([0,1], forecast_days)
        })

        future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek

        forecast_units = model.predict(
            future_df[["Campaign","Holiday","DayOfWeek"]]
        )

        unit_price = df_med["Unit_Price"].iloc[-1]

        revenue = forecast_units * unit_price

        temp_df = pd.DataFrame({
            "Medicine": med,
            "Units": forecast_units,
            "Revenue": revenue
        })

        all_forecasts.append(temp_df)

    all_forecasts = pd.concat(all_forecasts)


    # ---------- Summary ----------
    units_summary = all_forecasts.groupby("Medicine")["Units"].sum()
    revenue_summary = all_forecasts.groupby("Medicine")["Revenue"].sum()

    top_units = units_summary.idxmax()
    top_revenue = revenue_summary.idxmax()


    # ---------- Insights ----------
    st.subheader("Key Insights")

    st.write(f"Top medicine by units sold: **{top_units}**")
    st.write(f"Top medicine by revenue: **{top_revenue}**")


    # ---------- Units Graph ----------
    st.subheader("Forecasted Units Sold per Medicine")

    plt.figure()

    units_summary.plot(kind="bar")

    plt.ylabel("Units Sold")
    plt.xlabel("Medicine")
    plt.title("Forecasted Units Sold per Medicine")

    st.pyplot(plt)


    # ---------- Revenue Graph ----------
    st.subheader("Forecasted Revenue per Medicine")

    plt.figure()

    revenue_summary.plot(kind="bar")

    plt.ylabel("Revenue")
    plt.xlabel("Medicine")
    plt.title("Forecasted Revenue per Medicine")

    st.pyplot(plt)