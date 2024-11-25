# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:24:49 2024

@author: Misile Kunene

Autoregressive model
"""
# Libraries to be used 

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# 1. Data preparation
client = MongoClient(host = "localhost", port = 27017)
db = client["air-quality"]
nairobi = db["nairobi"]

# The wrangle function
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read data into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample to 1hr window. This will give us a Time Series data
    y = df["P2"].resample("1H").mean().fillna(method='ffill')

    return y

# Check the data
y = wrangle(nairobi)
y.head()

# 2. Exploratory data analysis (EDA)
# The ACF and PACF to check the autocorrelation
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax = ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")

fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax = ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")

# 3. Split the data
cutoff_test = int(len(y)*0.95)

y_train = y[:cutoff_test]
y_test = y[cutoff_test:]

# 4. Build the model
# Baseline model
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

# Iterate
model = AutoReg(y_train, lags = 26).fit()

# Make predictions
y_pred = model.predict().dropna()
training_mae = mean_absolute_error(y_train.iloc[26:], y_pred)
print("Training MAE:", training_mae)


# Check and visualize the residuals
y_train_resid = model.resid
y_train_resid.tail()

# Plot
fig, ax = plt.subplots(figsize=(15, 6))
y_train_resid.plot(ylabel = "Residual Value", ax = ax)

# Histogram
y_train_resid.hist()
plt.xlabel("Residual Valeu")
plt.ylabel("Frequency")
plt.title("AR(26), Distribution of Residuals")

# Check the ACF for the residuals
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax = ax)

# Evaluate the model
y_pred_test = model.predict(y_test.index.min(), y_test.index.max())
test_mae = mean_absolute_error(y_test, y_pred_test)
print("Test MAE:", test_mae)

# Make predictions
df_pred_test = pd.DataFrame(
    {"y_test": y_test, "y_pred": y_pred_test}, index=y_test.index
)

# Visualize predictions
fig = px.line(df_pred_test, labels={"value": "P2"})
fig.show()

# This model will behave badly, so we use the Walk-Forward-Validation technique
# WFV

%%capture

y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags = 26).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])
    
# Evaluate
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))

# Communicate the results
print(model.params)

# df_pred_test = pd.Series(
    {
    "y_test": y_test, "y_pred_wfv": y_pred_wfv
    }
)
fig = px.line(df_pred_test, labels = {"value": "PM2.5"})
fig.show()
























