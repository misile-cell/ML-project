# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:15:44 2023

@author: Misile 
"""
#Data visialization

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sklearn.linear_model

# Load the data for analysis

data = pd.read_csv("C:\Users\User\Chapter 2 - housing.csv")

# Heading
data.head()

# See the heads/features of the data
data.info() # we have 10 columns & 20640 rows, 9 of the columns are floats
# & 1 oject

# Check the catergories in the ocean proximity column
data["ocean_proximity"].value_counts()

# We get a summary of the data
X = data.describe()

# Plot the data
data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
          s=data["population"]/100, label="population", figsize=(10,7), 
          c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, ) 
plt.legend()

#Find the correlation matrix to understand the relationship etwee the variales
corr_matrix = data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# We see now that the media house value is correlated to median income

# Creating new attributes

# What we really want to understand  is the number of rooms per household. Similarly, 
# the total number of bedrooms by itself is not very useful: we want
# to compare it to the number of rooms

data["rooms_per_household"] = data["total_rooms"]/data["households"] 
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"] 
data["population_per_household"] = data["population"]/data["households"]

# Check the correlation matrix and you'll see that the number of bedrooms per room 
# is correlated with the median house value!

##########################################################################
# Prepare the Data for Machine Learning Algorithms

# Create a Test Set

# Create a test set of the data ~ 20%

#def split_train_test(dataset, test_ratio):    
#    shuffled_indices = np.random.permutation(len(dataset))    
#    test_set_size = int(len(dataset) * test_ratio)    
#    test_indices = shuffled_indices[:test_set_size]    
#    train_indices = shuffled_indices[test_set_size:]    
#    return dataset.iloc[train_indices], dataset.iloc[test_indices]

#train_data, test_data = split_train_test(data, 0.2)
#train_data.head()


# This code will always give different values, How to fix???
# The new test set will contain 20% of the new instances, 
# but it will not contain any instance that was previously 
# in the training set

#from zlib import crc32

#def test_set_check(identifier, test_ratio):    
#    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


#def split_train_test_by_id(dataset, test_ratio, id_column):    
 #   ids = dataset[id_column]
#    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))    
#    return dataset.loc[~in_test_set], dataset.loc[in_test_set] 

# Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways.

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# What if the dataset is not large? >>>  stratified sampling
# To decide on the size of the tratums, check the histogram first
# most median income values are clustered around 1.5 to 6

data["median_income"].hist(bins=50)

# In this case we can have 5 catergories: [0, 1.5];[1.5,3];[3,4.5];[4.5,6],[6,infty]

data["income_cat"] = pd.cut(data["median_income"], 
       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],                            
       labels=[1, 2, 3, 4, 5])

# Check the histogram
# Now you are ready to do stratified sampling based on the income category. 
# For this you can use Scikit-Learn’s StratifiedShuffleSplit class:

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set=data.loc[test_index]
    
# Check if the income category proportions are in line with the test set 
#(histogram):
strat_test_set["income_cat"].value_counts() / len(strat_test_set) # Confirmed 
  
# Now we remove the income_cat attribute so the data is back to its 
# original state 

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
    
    
data = strat_train_set.drop("median_house_value", axis=1) 
data_labels = strat_train_set["median_house_value"].copy()


# Discover and Visualize the Data to Gain Insights
 
    # Make a copy of the train data
 
#data = strat_train_set.copy()

# Data Cleaning


# For missing values, we can i) get rid of the whole feature ii) get rid of 
# the row or iii) fill the missing value with mean or median.

#median_strat_train_set = strat_train_set["total_bedrooms"].median()  # option 3 
#strat_train_set["total_bedrooms"].fillna(median_strat_train_set, inplace=True)

# Scikit-Learn provides a handy class to take care of missing values: 
# SimpleImputer. Here is how to use it. First, you need to create a 
# SimpleImputer instance, specifying that you want to replace each attribute’s
# missing values with the median of that attribute:

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # SimpleImputer instance

data_num = data.drop("ocean_proximity", axis=1) # Remove catergorical variale

# This will only e applied on the num (numerical) values
# fit the imputer instance to the training data 

#imputer.fit(strat_train_set_num)

#imputer.statistics_ # check the imputer statistic_ == strat_train_set_num.median().values

# Handling Text and Categorical Attribute
# Earlier we left out the categorical attribute ocean_proximity because it is 
# a text attribute so we cannot compute its median


# The OneHotEncoder

#data_cat = data[["ocean_proximity"]]

#from sklearn.preprocessing import OneHotEncoder
#cat_encoder = OneHotEncoder()
#data_cat_1hot = cat_encoder.fit_transform(data_cat)

# Once again, you can get the list of categories using the encoder’s 
# categories_ instance variable:
    
#cat_encoder.categories_
    
# Custom Transformers: for tasks such as custom cleanup operations or 
# combining specific attributes. We make sure that we bring the values of attriutes 
# within a certain interval

# Feature Scaling: With few exceptions, Machine Learning algorithms don’t 
# perform well when the input numerical attributes have very different scales.
# There are two common ways to get all attributes to have the same scale: 
# (1) min-max scaling -  normalization - and (2) standardization [(X - mu)/std].

# Transformation Pipelines: Scikit-Learn provides the Pipeline() class to help
#  with such sequences of transformations.

# (1) Pipeline for the numerical attributes:

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([        
        ('imputer', SimpleImputer(strategy="median")),        
        ('std_scaler', StandardScaler()),    
    ])
# The Pipeline constructor takes a list of name/estimator pairs defining a 
#sequence of steps:     ['imputer', 'attribs_adder', 'std_scaler'] ad the 
# last one should be a trasformer
#  numerical training data 
#num_tr_data = num_pipeline.fit_transform(strat_train_set_num)

   # (2) Catergorical: OneHotEncoder() 
    
 # This can be done in just one simple step as follows:

#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
num_attribs = list(data_num) 
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([        
        ("num", num_pipeline, num_attribs),        
        ("cat", OneHotEncoder(), cat_attribs),    
        ])
    
data_prepared = full_pipeline.fit_transform(data)    

# Linear Regression
    
from sklearn.linear_model import LinearRegression
 
lin_reg = LinearRegression() 
lin_reg.fit(data_prepared, data_labels)   
 
#  Let’s try it out on a few instances from the training set:   
    
some_data = data.iloc[:5]
some_labels = data_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))    
    

# Error 

from sklearn.metrics import mean_squared_error 
data_predictions = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(data_labels, data_predictions) 
lin_rmse = np.sqrt(lin_mse) 
lin_rmse
    
  # 68161.226444332  
  #209375.74268037, 315154.78319184, 210238.27856353,  55902.61573275,
      #183416.68718873
# 


