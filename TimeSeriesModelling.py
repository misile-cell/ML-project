# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:09:43 2024

@author: Misile
"""
# Retrieving Time Series data from Mongo Database

# Import important libraries

from pprint import PrettyPrinter

import pandas as pd
from IPython.display import VimeoVideo
from pymongo import MongoClient

# Initiate pretty printer
pp = PrettyPrinter(indent = 2)

# 1. Data preparation
# Connect to the database as a client
client = MongoClient(host = "localhost", port = 27017)

# Explore the databases that are available
pp.pprint(list(client.list_databases()))

# Choose a database that you want to work with
db = client["air-quality"]

# Check the list of collections in your database
for c in db.list_collections():
    print(c["type"])
    
# Select collection
nairobi = db["nairobi"]

# Count the number of documents
nairobi.count_documents({})

# Retrieve and inspect one document from the collection
result = nairobi.find_one({})
pp.pprint(result)

# Find groups using the distinct method
nairobi.distinct("metadata.site")

print("Documents from site 6:", nairobi.count_documents({"metadata.site": 6})
)
print("Documents from site 29:", nairobi.count_documents({"metadata.site": 29})
)

# Find the number of readings in each group
result = nairobi.aggregate(
    [{"$group": {"_id": "$metadata.site", "count": {"$count": {}}} }]
)
pp.pprint(list(result))

# Determine the types of measurements taken 
nairobi.distinct("metadata.measurement")

# Retrieve the readings using the find method
result = nairobi.find({"metadata.measurement": "P2"}).limit(3)
pp.pprint(list(result))

# Use aggregate to find the number of readings for each type
result = nairobi.aggregate(
    [   {"$match": {"metadata.site": 6}},
        
        {"$group": {"_id": "$metadata.measurement", "count": {"$count": {}}} }]
)
pp.pprint(list(result))

# Choose the features you want to work with 
result = nairobi.find(
    {"metadata.site": 29, "metadata.measurement": "P2"},
    projection = {"P2": 1, "timestamp": 1, "_id": 0}
)
pp.pprint(list(result.next()))

# Create a DataFrame
df = pd.DataFrame(result).set_index("timestamp")
df.head()

######################################################################
# 1. Building the moving average model

# Important libraries
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pytz
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Data preparation
client = MongoClient(host = "localhost", port = 27017)
db = client["air-quality"]
nairobi = db["nairobi"]

# Define the wrangle function for data cleaning purposes
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    df = pd.DataFrame(results).set_index("timestamp")
    
    # Localize timezone to Nairobi Timezone
    
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")
    
    # Remove outliers
    df = df[df["P2"] < 500]
    
    # Resample on 1 H window and use the forward fill method to fill missing values
    
    # Check if there are any null values using
    # df["P2"].resample("1H").mean().isnull().sum
    
    # Deal with the null values using
    # df["P2"].resample("1H").mean().fillna(method = "ffill")
    # Change to a dataframe
    df = df["P2"].resample("1H").mean().fillna(method = "ffill").to_frame()
    
    # Add the lag values
    df["P2.L1"] = df["P2"].shift(1)
    
    # Drop NaN row
    df.dropna(inplace = True)
    
    return df

# Read the nairobi collection using the wrangle function
df = wrangle(nairobi)
df.head()

# Create the boxplot to identify outliers
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind = "box", vert = False, title = "Distribution of P2 Readings", ax = ax)

# Create the time series plot
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel = "Time", ylabel = "PM2.5", title = "PM2 Time Series", ax = ax)

# Plot the rolling average using 168 hour window (weekly)
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].rolling(168).mean().plot(ax=ax, ylabel = "PM2.5", title = "Weekly Rolling Average");

# Check the correlation
df.corr()

# Create a scatterplot to investigate any correlation
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x = df["P2.L1"], y = df["P2"])
ax.plot([0,120], [0,120], linestyle = "--", color = "orange")
plt.xlabel("P2.L1")
plt.ylabel("P2")
plt.title("PM2.5 Autocorrelation")

# Split the data
target = "P2"
y = df[target]
X = df.drop(columns = target)

# Train-test data 
cutoff = int(len(X)*0.8)

X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

# 3.Buld the model
y_train_mean = y_train.mean()

y_pred_baseline = [y_train_mean]*len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2))


model = LinearRegression()

model.fit(X_train, y_train)

y_pred_training = pd.Series(model.predict(X_train))
y_pred_training.head()

training_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print("Training MAE:", round(training_mae, 2))
print("Test MAE:", round(test_mae, 2))

# 4. Communicate the results
intercept = model.intercept_
coefficient = model.coef_

print(f"P2 = {intercept} + ({coefficient} * P2.L1)")


df_pred_test = pd.DataFrame(
    {"y_test": y_test,
     "y_pred": model.predict(X_test)
    }
)
df_pred_test.head()

fig = px.line(df_pred_test, labels = {"value": "P2"})
fig.show()

































































