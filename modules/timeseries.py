import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from google.colab import files

print("📂 Upload your Excel file (must contain a date column and a numeric value column)")
uploaded = files.upload()

for file_name in uploaded.keys():
    df = pd.read_excel(file_name)
    print(f"✅ File '{file_name}' loaded.")

print("\n📄 Preview of your data:")
display(df.head())

original_columns = df.columns.tolist()
normalized_columns = {col.lower().strip(): col for col in df.columns}

print("\nAvailable columns:")
for col in original_columns:
    print(f"  - {col}")

def choose_column(prompt, dtype='any'):
    while True:
        selected = input(prompt + " (copy/paste or type exactly): ").lower().strip()
        if selected in normalized_columns:
            column = normalized_columns[selected]
            if dtype == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[column]):
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                except:
                    print("❌ Could not convert to datetime. Try another column.")
                    continue
            if dtype == 'numeric' and not pd.api.types.is_numeric_dtype(df[column]):
                print("❌ This column is not numeric. Try another.")
                continue
            return column
        else:
            print("❌ Column not found. Try again.")

date_col = choose_column("🕒 Enter the column name for date", dtype='datetime')

def choose_numeric_column(prompt):
    while True:
        selected = input(prompt + " (copy/paste or type exactly): ").lower().strip()
        if selected in normalized_columns:
            column = normalized_columns[selected]
            # Try converting to numeric, coercing errors to NaN
            converted = pd.to_numeric(df[column], errors='coerce')
            # Check if column has any numeric values
            if converted.notna().sum() == 0:
                print("❌ No numeric data found in this column. Try another.")
                continue
            # Show non-numeric cells
            non_numeric_mask = converted.isna() & df[column].notna()
            if non_numeric_mask.any():
                print(f"⚠️ Warning: Found non-numeric values in '{column}' at rows:")
                print(df.loc[non_numeric_mask, column])
                # Replace non-numeric with NaN
                df.loc[non_numeric_mask, column] = np.nan
            else:
                print(f"✅ All values in column '{column}' are numeric or missing.")
            # Update dataframe column to numeric dtype
            df[column] = pd.to_numeric(df[column], errors='coerce')
            return column
        else:
            print("❌ Column not found. Try again.")

value_col = choose_numeric_column("📊 Enter the column name for values")


df = df.dropna(subset=[date_col])
df.set_index(pd.to_datetime(df[date_col]), inplace=True)
ts = df[value_col].copy()

def handle_missing(ts, method='ffill'):
    if method == 'drop':
        return ts.dropna()
    elif method == 'ffill':
        return ts.fillna(method='ffill')
    elif method == 'interpolate':
        return ts.interpolate()
    else:
        raise ValueError("Method must be 'drop', 'ffill', or 'interpolate'.")

# Outlier handling
def handle_outliers(ts, z_thresh=3.0, method='median'):
    z_scores = (ts - ts.mean()) / ts.std()
    outliers = np.abs(z_scores) > z_thresh
    if method == 'median':
        ts[outliers] = ts.median()
    elif method == 'rolling':
        ts[outliers] = ts.rolling(window=7, center=True).mean()[outliers]
    return ts

# ADF Test
def adf_test(ts):
    result = adfuller(ts.dropna())
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "is_stationary": result[1] <= 0.05
    }

# Forecasting
def forecast_series(ts, days=30, m=7):
    model = auto_arima(ts, seasonal=True, m=m, trace=False, suppress_warnings=True)
    forecast = model.predict(n_periods=days)
    future_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')
    forecast_df = pd.DataFrame({"Forecast": forecast}, index=future_index)
    return forecast_df, model

# Plotting
def plot_series(ts, title):
    plt.figure(figsize=(10, 4))
    plt.plot(ts, label=value_col)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

def decompose_series(ts, period=7):
    return seasonal_decompose(ts, model="additive", period=period)

ts = handle_missing(ts, method='ffill')
ts = handle_outliers(ts, method='median')
plot_series(ts, f"{value_col} Over Time")

result = adf_test(ts)
print("\n📈 ADF Test:")
print(f"ADF Statistic: {result['ADF Statistic']:.4f}")
print(f"p-value: {result['p-value']:.4f}")
print("✅ Stationary" if result["is_stationary"] else "⚠️ Not Stationary")

try:
    decomposition = decompose_series(ts)
    decomposition.plot()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️ Could not decompose: {e}")

forecast_df, model = forecast_series(ts, days=30)
plot_series(forecast_df, "📉 Forecast for Next 30 Days")

forecast_df.to_excel("forecast_output.xlsx")
files.download("forecast_output.xlsx")
print("✅ Forecast exported to Excel.")
