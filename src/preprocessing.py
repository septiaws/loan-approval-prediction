#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.util as util
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# # 1.Load Config

# In[86]:


config_data = util.load_config()


# # 2. Load Dataset

# In[87]:


def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Return 3 set of data
    return train_set, valid_set, test_set


# In[88]:


train_set, valid_set, test_set = load_dataset(config_data)


# # 3. Encoding Categorical Data

# In[89]:


def cat_ohe_fit(config_data):
    for cat in config_data['predictors_categorical']:
        ohe_obj = OneHotEncoder(sparse = False,handle_unknown = 'ignore')
        # Fit ohe
        ohe_obj.fit(np.array(config_data["range_"+cat]).reshape(-1, 1))
        # Save ohe object
        util.pickle_dump(ohe_obj, config_data["ohe_"+cat+"_path"])


# In[90]:


cat_ohe_fit(config_data)


# In[91]:


def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_path: str) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Load ohe stasiun
    ohe_obj = util.pickle_load(ohe_path)

    # Transform variable of set data, resulting array
    features = ohe_obj.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    # Convert to dataframe
    column_name = [tranformed_column+"_"+s for s in ohe_obj.categories_[0]]
    features = pd.DataFrame(features.tolist(), columns = list(column_name))

    # Set index by original set data index
    features.set_index(set_data.index, inplace = True)

    # Concatenate new features with original set data
    set_data = pd.concat([features, set_data], axis = 1)

    # Drop stasiun column
    set_data.drop(columns = tranformed_column, inplace = True)

    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    # Return new feature engineered set data
    return set_data


# In[92]:


def cat_ohe_transform(set_data, config_data):
    set_data = set_data.copy()
    for cat in config_data['predictors_categorical']:
        set_data = ohe_transform(set_data,cat,config_data["ohe_"+cat+"_path"])
    return set_data


# In[93]:


train_set = cat_ohe_transform(train_set,config_data)
train_set


# In[94]:


valid_set = cat_ohe_transform(valid_set,config_data)
valid_set


# In[95]:


test_set = cat_ohe_transform(test_set,config_data)
test_set


# # 4. Balancing Data

# In[96]:


sns.histplot(data = train_set, x = "Loan_Status", hue = "Loan_Status")
plt.show()


# ### Balancing Data with SMOTE

# In[97]:


def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    x_sm, y_sm = sm.fit_resample(set_data.drop("Loan_Status", axis = 1), set_data["Loan_Status"])

    # Concatenate balanced data
    set_data_sm = pd.concat([x_sm, y_sm], axis = 1)

    # Return balanced data
    return set_data_sm


# In[98]:


train_set_sm = sm_fit_resample(train_set)


# In[99]:


sns.histplot(train_set_sm, x = "Loan_Status", hue = "Loan_Status")


# # 5. Label Encoder

# In[100]:


def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()

    # Fit le
    le_encoder.fit(data_tobe_fitted)

    # Save le object
    util.pickle_dump(le_encoder, le_path)

    # Return trained le
    return le_encoder


# In[101]:


le_encoder = le_fit(config_data["label_categories"], config_data["le_path"])


# In[102]:


def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    # Create copy of label_data
    label_data = label_data.copy()

    # Load le encoder
    le_encoder = util.pickle_load(config_data["le_path"])

    # If categories both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        # Transform label data
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")
    
    # Return transformed label data
    return label_data


# In[103]:


train_set_sm.Loan_Status = le_transform(train_set_sm.Loan_Status, config_data)


# In[104]:


valid_set.Loan_Status = le_transform(valid_set.Loan_Status, config_data)


# In[105]:


test_set.Loan_Status = le_transform(test_set.Loan_Status, config_data)


# In[106]:


x_train = {
    "SMOTE" : train_set_sm.drop(columns = "Loan_Status")
}

y_train = {
    "SMOTE" : train_set_sm.Loan_Status
}


# In[109]:


util.pickle_dump(x_train, "data/processed/x_train_feng.pkl")
util.pickle_dump(y_train, "data/processed/y_train_feng.pkl")

util.pickle_dump(valid_set.drop(columns = "Loan_Status"), "data/processed/x_valid_feng.pkl")
util.pickle_dump(valid_set.Loan_Status, "data/processed/y_valid_feng.pkl")

util.pickle_dump(test_set.drop(columns = "Loan_Status"), "data/processed/x_test_feng.pkl")
util.pickle_dump(test_set.Loan_Status, "data/processed/y_test_feng.pkl")

