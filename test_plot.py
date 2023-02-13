#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
#
# Copyright 2022 Ocean Protocol Foundation
# SPDX-License-Identifier: Apache-2.0
#
"""
=========================================================
 Air Quality in Catalonia
=========================================================

"""

# Code source: Ocean Protocol Foundation
# Modified for Air Quality in catalonia 
# Modified by Jose Giraldo
# License: BSD 3 clause


import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def get_input(local=False):
    if local:
        print("Reading local file aqi_data.csv")

        return "aqi_data.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename
    
def plot_pollutant(local=False):
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return
    
    pollutant = "O3"
    
    df = pd.read_csv(filename, header=0)
    # helper columns
    all_columns = df.columns
    value_columns = [f"0{+1+i}h" if i<9 else f"{1+i}h" for i in range(24)]
    id_columns = all_columns.difference(set(value_columns))
    
    # Filt df
    df_filt = df[df["CONTAMINANT"].isin([pollutant])]
    del df
    
    ## wide 2 long
    
    df_long = pd.melt(df_filt,id_vars=id_columns,value_vars=value_columns)

    df_long["hour"] = df_long["variable"].str.replace("h","")                         .str.replace("24","00")
    
    df_long["datetime"] = pd.to_datetime(df_long["DATA"] + " " + df_long["hour"],format="%d/%m/%Y %H")
    df_long = df_long.drop(columns=["variable","DATA","hour"])

    df_mean = df_long.groupby("datetime").mean()
    plt.title("Mean O3 concentration")
    plt.plot(df_mean.index,df_mean["value"].values,
             linewidth=0.5)
    plt.xlabel("date")
    plt.show()

    print(f"Data plotting success for df of shape {df_long.shape}")
    
    

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    plot_pollutant(local)

