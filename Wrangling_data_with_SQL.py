# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:34:05 2024

@author: Misile Kunene

Wrangling data with SQL

Ethiopia, Kenya, Burkina Faso, Nigeria, Malawi, Mozambique, Zimbabwe, 
Ghana



"""

# Important libraries
import sqlite3
import pandas as pd

# 1. Prepare data

# i) Connect to the SQL DataBase
%load_ext sql
%sql sqlite:////home/jovyan/nepal.sqlite

# ii) Connect to the Nepal SQLite3 database
%load_ext sql
%sql sqlite:////home/jovyan/nepal.sqlite

# 2. Explore the data
# Select rows and columns from the sql_schema table

%%sql
SELECT * # the asterick means 'all'
FROM sqlite_schema

# Select the name column from the sqlite_schema table, showing only rows where
# the type is "table"
%%sql
SELECT name
FROM sqlite_schema
WHERE type = "table"

# Select all columns from the id_map table, limiting your results to the first
# five rows.
%%sql
SELECT *
FROM id_map
LIMIT 5

# How many observations are in the id_map table? Use the count command to 
# find out.
%%sql
SELECT count(*)
FROM id_map
LIMIT 5

# What districts are represented in the id_map table? Use the distinct command
# to determine the unique values in the district_id column.
%%sql
SELECT distinct(district_id)
FROM id_map

# How many buildings are there in id_map table? Combine the count and distinct
# commands to calculate the number of unique values in building_id
%%sql

SELECT count(distinct(building_id))
FROM id_map

#  For our model, we'll focus on Gorkha (district 4). Select all columns that
# from id_map, showing only rows where the district_id is 4 and limiting your
# results to the first five rows.
%%sql
SELECT *
FROM id_map
WHERE district_id = 4
LIMIT 5

# How many observations in the id_map table come from Gorkha? Use the count
# and WHERE commands together to calculate the answer.
%%sql
SELECT count(*)
FROM id_map
WHERE district_id = 4

#  How many buildings in the id_map table are in Gorkha? Combine the count and
# distinct commands to calculate the number of unique values in building_id, 
# considering only rows where the district_id is 4.
%%sql
SELECT count(distinct(building_id)) AS unique_building_gorkha
FROM id_map
WHERE district_id = 4

# Select all the columns from the building_structure table, and limit your
# results to the first five rows.
%%sql
SELECT *
FROM building_structure
LIMIT 5

# How many building are there in the building_structure table? Use the count
# command to find out
%%sql
SELECT count(*)
FROM building_structure

#  There are over 200,000 buildings in the building_structure table, but how
# can we retrieve only buildings that are in Gorkha? Use the JOIN command to 
# join the id_map and building_structure tables, showing only buildings where 
# district_id is 4 and limiting your results to the first five rows of the new table.
%%sql
SELECT *
FROM id_map AS i
JOIN building_structure AS s ON i.building_id = s.building_id
WHERE district_id = 4
LIMIT 5

# Use the distinct command to create a column with all unique building IDs in 
# the id_map table. JOIN this column with all the columns from the 
# building_structure table, showing only buildings where district_id is 4 and 
# limiting your results to the first five rows of the new table.
%%sql
SELECT distinct(i.building_id),
    i.*
FROM id_map AS i
JOIN building_structure AS s ON i.building_id = s.building_id
WHERE district_id = 4
LIMIT 5

#  How can combine all three tables? Using the query you created in the last 
# task as a foundation, include the damage_grade column to your table by adding 
# a second JOIN for the building_damage table. Be sure to limit your results
# to the first five rows of the new table.
# Create an alias for a column or table using the AS command in SQL.
%%sql
SELECT distinct(i.building_id) AS b_id,
    i.*,
    d.damage_grade
FROM id_map AS i
JOIN building_structure AS s ON i.building_id = s.building_id
JOIN building_damage AS d ON i.building_id = d.building_id
WHERE district_id = 4
LIMIT 5




conn = sqlite3.connect("/home/jovyan/nepal.sqlite")

query = """
SELECT distinct(i.building_id) AS b_id,
    i.*,
    d.damage_grade
FROM id_map AS i
JOIN building_structure AS s ON i.building_id = s.building_id
JOIN building_damage AS d ON i.building_id = d.building_id
WHERE district_id = 4

"""
print(query)

# Read in a dataframe using pandas
df = pd.read_sql(query, conn, index_col = "b_id")

df.head()
















































