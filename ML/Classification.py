# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:13:37 2023

@author: Misile
"""

# Classificatio

# In this chapter, we are going to use Machine Learning Algorithms 
# (Classifiers) to classify the mnist data set

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sklearn.linear_model

# 1. Load the data

mnist_train = pd.read_csv("C:\Users\Python\mnist_train.csv")
mnist_test = pd.read_csv("C:\Users\Python\mnist_test.csv")



