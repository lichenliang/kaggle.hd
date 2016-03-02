
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
df_train = pd.read_csv('../../Data/train.csv')
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


stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
import re
from nltk.stem.porter import *
stemmer = PorterStemmer()

def str_stem(s):
    if isinstance(s, str):
        try:
            s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
            s = s.lower()
            s = s.replace("  "," ")
            s = s.replace(",","") #could be number / segment later
            s = s.replace("$"," ")
            s = s.replace("?"," ")
            s = s.replace("-"," ")
            s = s.replace("//","/")
            s = s.replace("..",".")
            s = s.replace(" / "," ")
            s = s.replace(" \\ "," ")
            s = s.replace("."," . ")
            s = re.sub(r"(^\.|/)", r"", s)
            s = re.sub(r"(\.|/)$", r"", s)
            s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
            s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
            s = s.replace(" x "," xbi ")
            s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
            s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
            s = s.replace("*"," xbi ")
            s = s.replace(" by "," xbi ")
            s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
            s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
            s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
            s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
            s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
            s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
            s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
            s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
            s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
            s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
            s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
            s = s.replace(" v "," volts ")
            s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
            s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
            s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
            s = s.replace("  "," ")
            s = s.replace(" . "," ")
            #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
            s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
            s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

            s = s.lower()
            s = s.replace("toliet","toilet")
            s = s.replace("airconditioner","air conditioner")
            s = s.replace("vinal","vinyl")
            s = s.replace("vynal","vinyl")
            s = s.replace("skill","skil")
            s = s.replace("snowbl","snow bl")
            s = s.replace("plexigla","plexi gla")
            s = s.replace("rustoleum","rust-oleum")
            s = s.replace("whirpool","whirlpool")
            s = s.replace("whirlpoolga", "whirlpool ga")
            s = s.replace("whirlpoolstainless","whirlpool stainless")
            return s
        except:
            return s
    else:
        return "null"



######################
## Pre-process Data ##
######################
print("Pre-process data...")


## clean text
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))

df_all_test['search_term'] = df_all_test['search_term'].map(lambda x:str_stem(x))
df_all_test['product_title'] = df_all_test['product_title'].map(lambda x:str_stem(x))
df_all_test['product_description'] = df_all_test['product_description'].map(lambda x:str_stem(x))
df_all_test['brand'] = df_all_test['brand'].map(lambda x:str_stem(x))

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


