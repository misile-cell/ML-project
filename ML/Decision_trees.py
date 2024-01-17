# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Wed Jan  17 15:01:11 2024

@author: Misile

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

%matplotlib inline
# Delete warning 
warnings.simplefilter("ignore")

# Data
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# Model Evalutions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Data visualization
# The data is obtained from Kaggle, can easily be downloaded.
        
df_raw = pd.read_csv('C:/Users/mlungeli/Desktop/Python/final_test.csv')
df_raw.head()

df_raw['size'].value_counts()

# Use seaborn for the plots

sns.countplot(x=df_raw["size"])

sns.displot(df_raw["age"])

sns.displot(df_raw["weight"])

sns.displot(df_raw["height"])

sns.distplot(x=df_raw['age'])

# We check if the are any outliers, remove themif present

cleaned_dfs = []
zscore_threshold = 3
for size_type in df_raw['size'].unique():
    ndf = df_raw[['age', 'height', 'weight']][df_raw['size'] == size_type]
    zscore = np.abs((ndf - ndf.mean()) / ndf.std())
    ndf = ndf[(zscore < zscore_threshold).all(axis=1)]
    ndf['size'] = size_type
    cleaned_dfs.append(ndf)
    df_cleaned = pd.concat(cleaned_dfs)
    print(df_cleaned.head())
    
df_raw.isna().sum()

# Clean the data by putting the mean if there is no entry

df_raw["age"] = df_raw["age"].fillna(df_raw['age'].median())
df_raw["height"] = df_raw["height"].fillna(df_raw['height'].median())
df_raw["weight"] = df_raw["weight"].fillna(df_raw['weight'].median())
    
# Replace class sizes with numeric values

df_raw['size'] = df_raw['size'].map({"XXS": 1,
                                     "S": 2,
                                     "M": 3,
                                     "L": 4,
                                     "XL": 5,
                                     "XXL": 6,
                                     "XXXL": 7})


# Check if it worked

df_raw

# Make the data to tell a story by doing some feature engineering
# Add the body mass index in the data (bmi)
# #Feature Engineering
df_raw["bmi"] = df_raw['weight'] / (df_raw['height']**2/100)
# df_raw["weight-squared"] = df_raw["weight"] * df_raw["weight"]
# df_raw["age-size"] = df_raw["age"] * df_raw["size"]


# Now we split the data into the features (independent variable) and target (dependent)

X = df_raw.drop(["size", "height", "weight"], axis=1) # Features
y = df_raw['size'] # Target

# Get the traing data sets and the testing data sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10)

len(X_train), len(X_test)

# Train the model

# Decision Tree classifier

def decision_tree_run(depth):
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(X_train,y_train)
    return tree_clf.score(X_train, y_train)

decision_tree_run(None)

accuracy = []
tree_depth = list(range(2, 100, 3))


for i in tree_depth:
    accuracy.append(decision_tree_run(i))
    
accuracy

# View the tree


from sklearn.tree import export_graphviz

export_graphviz(
 tree_clf,
 out_file=image_path("iris_tree.dot"),
 feature_names=iris.feature_names[2:],
 class_names=iris.target_names,
 rounded=True,
 filled=True
 )

# Make predictions

tree_clf.predict_proba(X_test)



# Naive Bayes
models = {"Naive Bayes" : GaussianNB()}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    # Set random seed
    np.random.seed(18)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit model to data
        model.fit(X_train, y_train)
        # Evaluate model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
        
    return model_scores

# Fit the model to the data

model_score = fit_and_score(models, X_train, X_test, y_train, y_test)
model_score

model_compare = pd.DataFrame(model_score, index=['accuracy'])
model_compare.T.plot.bar();





