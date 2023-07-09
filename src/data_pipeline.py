#!/usr/bin/env python
# coding: utf-8

# # 1. Load Required Libraries

# In[139]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
import joblib
import os
import yaml
import src.util as util


# # 2. Load Configuration File

# In[140]:


config = util.load_config()


# # 3. Load Dataset

# In[141]:


raw_dataset = pd.read_csv(config["dataset_path"])


# In[142]:


raw_dataset


# In[143]:


dataset = raw_dataset.drop(['Loan_ID'], axis=1)


# # 4. Data Validation

# ### 4.1 Null values

# In[144]:


dataset.isnull().sum()


# In[145]:


dataset[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = dataset[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']].fillna(dataset[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']].median())
dataset[['Gender', 'Married', 'Dependents', 'Self_Employed']] = dataset[['Gender', 'Married', 'Dependents', 'Self_Employed']].fillna("Unknown")


# In[146]:


dataset.isnull().sum()


# ### 4.2 Tipe Data

# In[147]:


dataset.dtypes


# ### 4.3 Range Data

# In[148]:


dataset.describe()


# ### 4.4 Dimensi Data 

# In[149]:


dataset.shape


# ### 4.5 Remove Duplicates

# In[150]:


def removeDuplicates(data):
    print(f"shape awal                    : {data.shape}, (#observasi, #fitur)")
    
     # Drop duplicate
    data = data.drop_duplicates()
    print(f"shape setelah drop duplikat   : {data.shape}, (#observasi, #fitur)")

    return data


# In[151]:


dataset = removeDuplicates(dataset)
dataset


# # 5. Data Defense

# In[152]:


def check_data(input_data, params):
    # Check data types
    assert input_data.select_dtypes("float").columns.to_list() == params["float64_columns"], "an error occurs in float64 column(s)."
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    assert input_data.select_dtypes("int").columns.to_list() == params["int64_columns"], "an error occurs in int64 column(s)."

    # Check range of data
    assert input_data["ApplicantIncome"].between(params["range_ApplicantIncome"][0], params["range_ApplicantIncome"][1]).sum() == len(input_data), "an error occurs in age ApplicantIncome."
    assert input_data["CoapplicantIncome"].between(params["range_CoapplicantIncome"][0], params["range_CoapplicantIncome"][1]).sum() == len(input_data), "an error occurs in amount CoapplicantIncome."
    assert input_data["LoanAmount"].between(params["range_LoanAmount"][0], params["range_LoanAmount"][1]).sum() == len(input_data), "an error occurs in LoanAmount range."
    assert input_data["Loan_Amount_Term"].between(params["range_Loan_Amount_Term"][0], params["range_Loan_Amount_Term"][1]).sum() == len(input_data), "an error occurs in Loan_Amount_Term range."
    assert input_data["Credit_History"].between(params["range_Credit_History"][0], params["range_Credit_History"][1]).sum() == len(input_data), "an error occurs in Credit_History range."
    assert set(input_data["Gender"]).issubset(set(params["range_Gender"])), "an error occurs in Gender range."
    assert set(input_data["Married"]).issubset(set(params["range_Married"])), "an error occurs in Married range."
    assert set(input_data["Dependents"]).issubset(set(params["range_Dependents"])), "an error occurs in Dependents range."
    assert set(input_data["Education"]).issubset(set(params["range_Education"])), "an error occurs in Education range."
    assert set(input_data["Self_Employed"]).issubset(set(params["range_Self_Employed"])), "an error occurs in Self_Employed range."
    assert set(input_data["Property_Area"]).issubset(set(params["range_Property_Area"])), "an error occurs in Property_Area range."
    assert set(input_data["Loan_Status"]).issubset(set(params["range_Loan_Status"])), "an error occurs in Loan_Status range."
   


# In[153]:


check_data(dataset, config)


# # 6. Data Splitting

# In[154]:


def splitInputOtput(data):
    x = data[config["predictors"]].copy()
    y = data[config["label"]].copy()
    return x,y


# In[155]:


x,y = splitInputOtput(dataset)


# In[156]:


dataset.columns


# In[157]:


x


# In[158]:


y.value_counts()


# In[159]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, stratify = y)


# In[160]:


x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test)


# In[161]:


util.pickle_dump(x_train, config["train_set_path"][0])
util.pickle_dump(y_train, config["train_set_path"][1])

util.pickle_dump(x_valid, config["valid_set_path"][0])
util.pickle_dump(y_valid, config["valid_set_path"][1])

util.pickle_dump(x_test, config["test_set_path"][0])
util.pickle_dump(y_test, config["test_set_path"][1])

