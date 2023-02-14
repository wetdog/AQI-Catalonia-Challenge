#!/usr/bin/env python
# coding: utf-8

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


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
    
def forecast_hourly(local=False):
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
    categorical_features =["month","year"]
    target = "value"

    # Filt df
    df_filt = df[df["CONTAMINANT"].isin([pollutant])]
    del df

    ## wide 2 long
    df_long = pd.melt(df_filt,id_vars=id_columns,value_vars=value_columns)
    df_long["hour"] = df_long["variable"].str.replace("h","").str.replace("24","00")
    df_long["datetime"] = pd.to_datetime(df_long["DATA"] + " " + df_long["hour"],format="%d/%m/%Y %H")
    df_long = df_long.drop(columns=["variable",
                                    "DATA",
                                    "hour",
                                    "GEOREFERENCIA",
                                    "NOM COMARCA",
                                    ])

    df_long = df_long.dropna()
    # sort values
    df_long = df_long.sort_values(by="datetime")
    monthly_o3 = df_long.groupby(pd.Grouper(key='datetime', freq='M')).agg({'value': 'mean'}).reset_index()

    # add datetime columns
    monthly_o3["month"] = monthly_o3["datetime"].dt.month
    monthly_o3["year"] = monthly_o3["datetime"].dt.month


    # split data
    train_df = monthly_o3[monthly_o3.datetime < "2019-01-01 00:00:00"]
    test_df = monthly_o3[monthly_o3.datetime > "2019-01-01 00:00:00"]

    # train model

    x_train = train_df[categorical_features]
    y_train = train_df[target]
    x_test = test_df[categorical_features]
    y_test = test_df[target]

    model = HistGradientBoostingRegressor(loss="squared_error",
                                        learning_rate=0.05,
                                            validation_fraction=0.15,
                                            max_iter=200)

    model.fit(x_train,y_train)
    print(f"Test score R2: {model.score(x_test,y_test)}")

    ## monthly predictions

    start_date = "2023/01/01 00:00:00"
    end_date = "2024/12/12 00:00:00"

    dates = pd.date_range(start=start_date,
                end=end_date,
                freq="M")

    df_forecast = pd.DataFrame(dates,columns=["datetime"])

    # add datime columns
    df_forecast["month"] = df_forecast["datetime"].dt.month
    df_forecast["year"] = df_forecast["datetime"].dt.year

    df_forecast["O3_forecast"] = model.predict(df_forecast[categorical_features])
    
    out = df_forecast

    filename = "forecastM.pickle" if local else "/data/outputs/result"
    with open(filename, "wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(out, pickle_file)
    
    

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    forecast_hourly(local)

