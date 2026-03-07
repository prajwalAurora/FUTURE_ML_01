import os
import subprocess
import sys
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib import pyplot as plt
from datetime import timedelta

# ----------------- Auto-install dependencies -----------------
try:
    import xgboost
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost
from xgboost import XGBRegressor

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    from statsmodels.tsa.arima.model import ARIMA

from sklearn.linear_model import LinearRegression

# ----------------- Global DataFrame -----------------
df = None

# ----------------- Generate Sample CSV -----------------
def generate_sample_csv(filename="sample_medical_data.csv"):
    global df
    dates = pd.date_range(start="2026-01-01", periods=60)
    medicines = ["Paracetamol", "Ibuprofen", "Amoxicillin", "Vitamin C",
                 "Aspirin", "Cetirizine", "Metformin", "Omeprazole"]
    data = []
    for med in medicines:
        demand = np.random.randint(50, 150, size=len(dates))
        campaign = np.random.choice([0, 1], size=len(dates))
        holiday = np.random.choice([0, 1], size=len(dates))
        price = np.random.uniform(20, 80, size=len(dates))
        for i in range(len(dates)):
            data.append([dates[i], med, demand[i], campaign[i], holiday[i], price[i]])
    df = pd.DataFrame(data, columns=["Date", "Medicine", "Units_Sold", "Campaign", "Holiday", "Unit_Price"])
    df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]
    df.to_csv(filename, index=False)

    medicine_list = df["Medicine"].unique().tolist()
    medicine_combo["values"] = medicine_list
    if medicine_list:
        medicine_combo.current(0)
    messagebox.showinfo("Sample CSV Generated", f"{filename} created and loaded with {len(medicine_list)} medicines!")

# ----------------- Load CSV -----------------
def load_csv():
    global df
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        df = pd.read_csv(filepath, parse_dates=["Date"])
        if "Unit_Price" not in df.columns:
            df["Unit_Price"] = np.random.uniform(20, 80, size=len(df))
        df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]
        medicine_list = df["Medicine"].unique().tolist()
        medicine_combo["values"] = medicine_list
        if medicine_list:
            medicine_combo.current(0)
        messagebox.showinfo("File Loaded", f"{os.path.basename(filepath)} loaded successfully!")

# ----------------- Automatic Observation -----------------
def generate_observation(forecast_df):
    obs = ""
    avg_units = forecast_df["Units_Sold"].mean()
    max_units = forecast_df["Units_Sold"].max()
    max_day = forecast_df.loc[forecast_df["Units_Sold"].idxmax(), "Date"].strftime("%Y-%m-%d")
    
    avg_revenue = forecast_df["Revenue"].mean()
    max_revenue = forecast_df["Revenue"].max()
    max_rev_day = forecast_df.loc[forecast_df["Revenue"].idxmax(), "Date"].strftime("%Y-%m-%d")
    
    if forecast_df["Units_Sold"].iloc[-1] > forecast_df["Units_Sold"].iloc[0]:
        obs += "Forecast shows an increasing demand trend.\n"
    else:
        obs += "Forecast shows a decreasing/stable demand trend.\n"
        
    obs += f"Peak units ({int(max_units)}) expected on {max_day}.\n"
    obs += f"Peak revenue (₹{max_revenue:.2f}) expected on {max_rev_day}.\n"
    obs += f"Average daily units: {int(avg_units)}, average daily revenue: ₹{avg_revenue:.2f}\n"
    
    if "Holiday" in forecast_df.columns:
        if forecast_df["Holiday"].sum() > 0:
            obs += f"{int(forecast_df['Holiday'].sum())} holiday days may affect demand.\n"
    
    return obs

# ----------------- Forecast Function -----------------
def forecast_demand(df, medicine, forecast_days=30, model_name="XGBoost"):
    df_med = df[df["Medicine"]==medicine].copy().sort_values("Date")
    df_med["DayOfWeek"] = df_med["Date"].dt.dayofweek
    features = ["Campaign", "Holiday", "DayOfWeek"]
    X = df_med[features]
    y = df_med["Units_Sold"]

    if model_name=="XGBoost":
        model = XGBRegressor(objective="reg:squarederror", n_estimators=200)
        model.fit(X, y)
        last_date = df_med["Date"].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Campaign": np.random.choice([0,1], size=forecast_days),
            "Holiday": np.random.choice([0,1], size=forecast_days)
        })
        forecast_df["DayOfWeek"] = [d.dayofweek for d in forecast_df["Date"]]
        forecast_df["Units_Sold"] = model.predict(forecast_df[features])

    elif model_name=="Linear Regression":
        lr_model = LinearRegression()
        lr_model.fit(np.arange(len(y)).reshape(-1,1), y)
        forecast_df = pd.DataFrame({
            "Date":[df_med["Date"].max() + timedelta(days=i) for i in range(1, forecast_days+1)]
        })
        forecast_df["Units_Sold"] = lr_model.predict(np.arange(len(y), len(y)+forecast_days).reshape(-1,1))

    elif model_name=="ARIMA":
        arima_model = ARIMA(y, order=(5,1,0))
        arima_res = arima_model.fit()
        forecast_values = arima_res.forecast(steps=forecast_days)
        forecast_df = pd.DataFrame({
            "Date":[df_med["Date"].max() + timedelta(days=i) for i in range(1, forecast_days+1)],
            "Units_Sold": forecast_values
        })

    forecast_df["Unit_Price"] = df_med["Unit_Price"].iloc[-1]
    forecast_df["Revenue"] = forecast_df["Units_Sold"] * forecast_df["Unit_Price"]

    combined_df = pd.concat([df_med[["Date","Units_Sold","Revenue"]], forecast_df[["Date","Units_Sold","Revenue"]]], ignore_index=True)
    return combined_df, forecast_df

# ----------------- GUI Functions -----------------
def run_single_forecast():
    global df
    if df is None:
        messagebox.showerror("No Data","Please generate or load a CSV first!")
        return
    med = medicine_combo.get()
    if med=="":
        messagebox.showerror("Select Medicine","Please select a medicine.")
        return
    try:
        days = int(forecast_days_entry.get())
    except:
        messagebox.showerror("Invalid Input","Forecast days must be integer")
        return
    model_name = model_option.get()
    combined_df, forecast_df = forecast_demand(df, med, days, model_name)

    # Graph
    plt.figure(figsize=(10,5))
    plt.plot(combined_df["Date"], combined_df["Units_Sold"], label="Units Sold", color="blue")
    plt.plot(combined_df["Date"], combined_df["Revenue"], label="Revenue", color="green")
    plt.axvline(df[df["Medicine"]==med]["Date"].max(), color="red", linestyle="--", label="Forecast Start")
    plt.title(f"Forecast for {med} ({model_name})")
    plt.xlabel("Date")
    plt.ylabel("Units / Revenue (₹)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Treeview
    for row in forecast_table.get_children():
        forecast_table.delete(row)
    for i,row in forecast_df.iterrows():
        forecast_table.insert("", "end", values=(row["Date"].strftime("%Y-%m-%d"), int(row["Units_Sold"]), f"₹{row['Revenue']:.2f}"))

    # Automatic Observation
    auto_obs = generate_observation(forecast_df)
    observation_text.delete("1.0", tk.END)
    observation_text.insert(tk.END, auto_obs)

    # Save CSV + Excel
    forecast_df["Observation"] = auto_obs
    forecast_df.to_csv(f"forecast_{med}.csv", index=False)
    forecast_df.to_excel(f"forecast_{med}.xlsx", index=False)
    messagebox.showinfo("Saved","Forecast saved as CSV and Excel!")

def run_multi_comparison():
    global df
    if df is None:
        messagebox.showerror("No Data","Please generate or load a CSV first!")
        return
    try:
        days = int(forecast_days_entry.get())
    except:
        messagebox.showerror("Invalid Input","Forecast days must be integer")
        return
    all_forecasts = pd.DataFrame()
    medicine_list = df["Medicine"].unique().tolist()
    for med in medicine_list:
        _, forecast_df = forecast_demand(df, med, days, model_option.get())
        forecast_df["Medicine"] = med
        all_forecasts = pd.concat([all_forecasts, forecast_df], ignore_index=True)

    # Aggregate comparison
    total_units = all_forecasts.groupby("Medicine")["Units_Sold"].sum().sort_values(ascending=False)
    total_revenue = all_forecasts.groupby("Medicine")["Revenue"].sum().sort_values(ascending=False)

    # Plots
    plt.figure(figsize=(10,5))
    plt.bar(total_units.index, total_units.values, color="skyblue")
    plt.title("Forecasted Units Sold per Medicine")
    plt.ylabel("Units Sold")
    plt.xlabel("Medicine")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.bar(total_revenue.index, total_revenue.values, color="lightgreen")
    plt.title("Forecasted Revenue per Medicine")
    plt.ylabel("Revenue (₹)")
    plt.xlabel("Medicine")
    plt.show()

    # Display ranking in Treeview
    for row in forecast_table.get_children():
        forecast_table.delete(row)
    for med in total_units.index:
        forecast_table.insert("", "end", values=(med, int(total_units[med]), f"₹{total_revenue[med]:.2f}"))

    # Automatic Observation
    obs = f"Top medicine by units: {total_units.idxmax()}\nTop medicine by revenue: {total_revenue.idxmax()}"
    observation_text.delete("1.0", tk.END)
    observation_text.insert(tk.END, obs)

    # Save CSV + Excel
    all_forecasts.to_csv("multi_medicine_forecast.csv", index=False)
    all_forecasts.to_excel("multi_medicine_forecast.xlsx", index=False)
    messagebox.showinfo("Saved","Multi-medicine forecast saved as CSV and Excel!")

# ----------------- GUI Layout -----------------
root = tk.Tk()
root.title("Medical Demand Forecasting GUI")
root.geometry("700x700")
root.configure(bg="#f0f0f0")

# Buttons
tk.Button(root,text="Generate Sample CSV", command=generate_sample_csv, bg="#28a745", fg="white", font=("Arial", 11, "bold")).grid(row=0,column=0,padx=10,pady=10)
tk.Button(root,text="Load CSV", command=load_csv, bg="#17a2b8", fg="white", font=("Arial", 11, "bold")).grid(row=0,column=1,padx=10,pady=10)
tk.Button(root,text="Run Single Medicine Forecast", command=run_single_forecast, bg="#ffc107", fg="black", font=("Arial", 11, "bold")).grid(row=1,column=0,padx=10,pady=5)
tk.Button(root,text="Run Multi-Medicine Comparison", command=run_multi_comparison, bg="#6f42c1", fg="white", font=("Arial", 11, "bold")).grid(row=1,column=1,padx=10,pady=5)

# Medicine selection
tk.Label(root,text="Select Medicine:", bg="#f0f0f0", font=("Arial",10,"bold")).grid(row=2,column=0,padx=10,pady=5)
medicine_combo = ttk.Combobox(root)
medicine_combo.grid(row=2,column=1,padx=10,pady=5)

# Forecast days
tk.Label(root,text="Forecast Days:", bg="#f0f0f0", font=("Arial",10,"bold")).grid(row=3,column=0,padx=10,pady=5)
forecast_days_entry = tk.Entry(root)
forecast_days_entry.insert(0,"30")
forecast_days_entry.grid(row=3,column=1,padx=10,pady=5)

# Model selection
tk.Label(root,text="Select Model:", bg="#f0f0f0", font=("Arial",10,"bold")).grid(row=4,column=0,padx=10,pady=5)
model_option = ttk.Combobox(root, values=["XGBoost","Linear Regression","ARIMA"])
model_option.grid(row=4,column=1,padx=10,pady=5)
model_option.current(0)

# ----------------- Forecast Table -----------------
forecast_table = ttk.Treeview(root, columns=("Medicine/Date","Units_Sold","Revenue"), show="headings", height=12)
forecast_table.heading("Medicine/Date",text="Medicine / Date")
forecast_table.heading("Units_Sold",text="Forecasted Units Sold")
forecast_table.heading("Revenue",text="Forecasted Revenue")
forecast_table.grid(row=5,column=0,columnspan=2,padx=10,pady=10)

# ----------------- Observation / Notes -----------------
tk.Label(root, text="Observation / Notes:", bg="#f0f0f0", font=("Arial",10,"bold")).grid(row=6, column=0, padx=10, pady=5)
observation_text = tk.Text(root, height=8, width=60)
observation_text.grid(row=7, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()