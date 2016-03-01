
"""
__file__

    preprocess.py

__description__

    This file preprocesses data.

__author__

    Chenglong Chen
    
"""

import sys
import cPickle
import numpy as np
import pandas as pd
from nlp_utils import clean_text, pos_tag_text
sys.path.append("../")
from param_config import config

###############
## Load Data ##
###############
print("Load data...")
df_train = pd.read_csv('../../Data/train.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../../Data/product_descriptions.csv')
df_attr = pd.read_csv('../../Data/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]
df_all = pd.merge(df_train, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')


df_test = pd.read_csv('../../Data/test.csv')
df_all_test = pd.merge(df_test,df_pro_desc,how='left',on='product_uid')
df_all_test = pd.merge(df_all_test,df_brand,how='left',on='product_uid')
print("Done.")


######################
## Pre-process Data ##
######################
print("Pre-process data...")


## clean text


###############
## Save Data ##
###############
print("Save data...")

df_all = df_all.rename(columns={'search_term': 'query'})
with open(config.processed_train_data_path, "wb") as f:
    cPickle.dump(df_all, f, -1)
df_all_test = df_all_test.rename(columns={'search_term': 'query'})
with open(config.processed_test_data_path, "wb") as f:
    cPickle.dump(df_all_test, f, -1)

print("Done.")


