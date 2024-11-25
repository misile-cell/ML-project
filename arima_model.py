# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:00:28 2024

@author: Misile Kunene

# Fitting an ARIMA model
"""
# Libraries needed

import inspect
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# 1. Prepare the data
client = MongoClient(host = "localhost", port = 27017)
db = client["air-quality"]
nairobi = db["nairobi"]

# The wrangle function: customize function with the resample_rule argment
def wrangle(collection, resample_rule = "1H"):
 
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read results into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample and forward-fill
    y = df["P2"].resample("1H").mean().fillna(method = "ffill")

    return y

# Check function is givin us what we expect
wrangle(nairobi,resample_rule = "1D")

# 2. Exploratory data analysis (EDA)
# Check the autocorrelation using the acf
fig, ax = plt.subplots(figsize = (15, 6))
plot_acf(y, ax = ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coeffiecient")
plt.title("Autocorrelation")

# Check the partial autocorrelation using the pacf 
fig, ax = plt.subplots(figsize = (15, 6))
plot_pacf(y, ax = ax)
plt.xlabel("Lag[hours]")
plt.ylabel("Coerrelation Coefficient")
plt.title("PACF")

# Split the data
y_train = y.loc['2018-10']
y_test = y.loc["2018-11-01": "2018-11-01"]

# 3. Build the model
# Baseline model
y_train_mean = y_train.mean()
y_train_baseline = [y_train_mean]*len(y_train)
mae_baseline = mean_absolute_error(y_train, y_train_baseline)
print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

# 4. Parameter tuning (order of the arima model), that is , (p,0,q)
p_params = range(0, 25, 8)
q_params = range(0, 3, 1)

# Model optimization using the Grid Search method
# Create dictionary to store MAEs
mae_grid = dict()
# Outer loop: Iterate through possible values for `p`
for p in p_params:
    # Create key-value pair in dict. Key is `p`, value is empty list.
    mae_grid[p] = list()
    # Inner loop: Iterate through possible values for `q`
    for q in q_params:
        # Combination of hyperparameters for model
        order = (p, 0, q)
        # Note start time
        start_time = time.time()
        # Train model
        model = ARIMA(y_train, order = order).fit()
        # Calculate model training time
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict()
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        # Append MAE to list in dictionary
        mae_grid[p].append(mae)

print()
print(mae_grid)

# 5. Visualize the optimum parameters
mae_df = pd.DataFrame(mae_grid)
mae_df.round(4)

# The heatmap plot
sns.heatmap(mae_df, cmap = "Blues")
plt.xlabel("p values")
plt.ylabel("q value")
plt.title("ARMA Grid Search (Criterion: MAE)")

# Visualize the residuals
fig, ax = plt.subplots(figsize=(15, 12))
model.plot_diagnostics(fig = fig)

# 6. Model validation using the walk forward validation method
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = ARIMA(history, order = (8, 0, 1)).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])
    
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))
    
    
# 7. Communicate the result by plotting the predictions vs the test data
df_predictions = pd.DataFrame(
        {
            "y_test": y_test, "y_pred_wfv": y_pred_wfv
        }
    )

fig = px.line(df_predictions, labels = {"value": "PM2.5"})
fig.show()
    
    
    
    
    
    
    
    
    





































































































