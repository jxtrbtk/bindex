#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import json

import numpy as np
import pandas as pd

import lib
import lib.features

if __name__ == "__main__":
    root_folder = lib.features.STORAGE
    i,ie = 0,0
    data = []
    for symbol in os.listdir(root_folder):
        symbol_folder = os.path.join(root_folder,symbol)
        for ts in os.listdir(symbol_folder):
            ts_folder = os.path.join(symbol_folder,ts)

            try:
                if os.path.exists(os.path.join(ts_folder, "klines_5m_delayed_3d.json")):
                    j_data = lib.features.build_source_dataset(ts_folder)
                    data_x = lib.features.build_x(j_data)

                    j_data = lib.features.build_destination_dataset(ts_folder)
                    j_klines_5m_delayed_3d = j_data[-1]
                    j_time = j_data[0]
                    h,l = lib.features.get_future_high_low(j_klines_5m_delayed_3d, j_time)        
                    data_x["target_high"] = h
                    data_x["target_low"] = l
                    a,b = lib.features.get_optimal_ask_bid(data_x, j_klines_5m_delayed_3d, j_time)
                    data_x["target_ask"] = a
                    data_x["target_bid"] = b

                    data.append(data_x)

                    i = i + 1
            
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("extractor error:", exc_type, exc_tb.tb_lineno, str(e))
                ie = ie + 1

            print(i, ie, ts_folder, a, b)


    df = pd.DataFrame(dtype=float).from_dict(data)
    df.to_csv("extract.csv", index=False)