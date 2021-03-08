#!/usr/bin/env python
# coding: utf-8

import os
import random
import json

import numpy as np
import pandas as pd

import sys
import traceback
import datetime
import time

import lib
import lib.features
import lib.wallet

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def extract_data():
    root_folder = lib.features.STORAGE
    i,ie,a,b = 0,0,0,0
    data = []
    for symbol in os.listdir(root_folder):
        symbol_folder = os.path.join(root_folder,symbol)
        if os.path.isdir(symbol_folder):
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

                print("{:04} {:04} {:>50} {:.04f} {:.04f}".format(i, ie, ts_folder, a, b))

    df = pd.DataFrame(dtype=float).from_dict(data)
    file_path = os.path.join(root_folder, "extract.csv")
    df.to_csv(file_path, index=False)
    
    return df 

def get_data_set(df):
    
    #impute NaN
    print("imputation")
    highest_non_inf = df.replace(np.Inf, np.nan).max()
    df = df.replace(np.Inf, highest_non_inf)
    lowest_non_inf = df.replace(-np.Inf, np.nan).min()
    df = df.replace(-np.Inf, lowest_non_inf)
    cols_to_impute = [c for c in df.columns if not c.startswith("target_") and not c.startswith("base_")]
    for c in cols_to_impute:
        df[c] = df[c].fillna(df[c].median())
    
    # remove outliers
    where0 = (df["target_ask"] == 0.0) | (df["target_bid"] == 0.0)
    where3 = (df["target_ask"] >= 3.0) | (df["target_bid"] >= 3.0)
    where  = where0 | where3
    df = df[~where]
    df = df.reset_index(drop=True)
    print("outliers {}".format(where.sum()))
    
    # remove "time DMZ"
    df["base_ap_time"] = pd.to_datetime(df["base_ap_time"], unit="s", utc=True)
    training_ratio = 0.2
    dmz_hours = 24*3
    df = df.sort_values(by="base_ap_time", ascending=False)
    df = df.reset_index(drop=True)
    df["base_split"] = None
    test_split_bundary = df.head(int(df.shape[0]*training_ratio))["base_ap_time"].min()
    where = (df["base_ap_time"] >= test_split_bundary)
    df.loc[where, "base_split"] = "test"
    training_split_bundary = test_split_bundary - datetime.timedelta(hours=dmz_hours)
    where = (df["base_ap_time"] < training_split_bundary)
    df.loc[where, "base_split"] = "train"
    df = df.drop(columns=["base_ap_time"])
    where = df["base_split"].isna() 
    df = df[~where]
    df = df.reset_index(drop=True)
    print("time DMZ {}".format(where.sum()))
    
    y_cols = [c for c in df.columns if c.startswith("base") or c.startswith("target")]
    x_cols = [c for c in df.columns if c not in y_cols]

    # build train and test set
    where_train = (df["base_split"] == "train")
    where_test = (df["base_split"] == "test")
    x_train = df.loc[where_train, x_cols].values
    y_train = df.loc[where_train, ["target_ask","target_bid"]].values
    x_test = df.loc[where_test, x_cols].values
    y_test = df.loc[where_test, ["target_ask","target_bid"]].values
    print("data set train {}/{} test {}/{}".format(
        x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test

def format_params(params, param_var, param_grid, param_format):
    for k,e in params.items():
        if k in param_var:
            if e > np.max(param_grid[k]): e = np.max(param_grid[k])  
            if e < np.min(param_grid[k]): e = np.min(param_grid[k])  
        params[k] = param_format[k].format(e)
        data_type = type(param_grid[k][0])
        if data_type == type(1):
            params[k] = int(params[k])
        if data_type == type(1.1):
            params[k] = float(params[k])
    return params

CACHE = {}
def compute_score(params, param_var, param_grid, param_format, x_train, x_test, y_train, y_test):
    global CACHE
    params = format_params(params, param_var, param_grid, param_format)
    if str(params) in CACHE.keys(): 
        score = CACHE[str(params)]
    else: 
        model = XGBRegressor(**params, use_label_encoder=False)
#         model.set_params(**params)
        model = model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                   early_stopping_rounds=100, verbose=False)
        y_pred = model.predict(x_test)
        score = mean_squared_error(y_test, y_pred)
        CACHE[str(params)] = score
    return score, params

def get_param_var(param_grid):
    return dict([(e, k[-1]) for e,k in param_grid.items() if len(k)>1] + \
     [(e, k[0]) for e,k in param_grid.items() if len(k)==1])

def find_best_param(param_grid, param_format, x_train, x_test, y_train, y_test):
    global CACHE 
    CACHE = {}
    param_var = [e for e,k in param_grid.items() if len(k)>1]
    best_score = 10000.0
    best_grid = {}
    max_iter = 100
    start = time.time()
    for step in range(10000):
        since = time.time()

        random.shuffle(param_var)
        key = param_var[0]
        print("ooo {} {} {}{}  (--{}--)   ".format(
            key, param_grid[key][-1], u"\u00B1", np.std(param_grid[key]), time.asctime() ), end="\r")

        params1 = get_param_var(param_grid)
        val1 = param_grid[key][-1]
        params1[key] = val1 
        score1, params1 = compute_score(params1, param_var, param_grid, param_format, x_train, x_test, y_train, y_test)
        print("xoo {} {} {}{}  (--{}--)   ".format(
            key, param_grid[key][-1], u"\u00B1", np.std(param_grid[key]), time.asctime() ), end="\r")

        params2 = get_param_var(param_grid)
        val2 = param_grid[key][-1] - np.std(param_grid[key])
        params2[key] = val2
        score2, params2 = compute_score(params2, param_var, param_grid, param_format, x_train, x_test, y_train, y_test)
        print("xxo {} {} {}{}  (--{}--)   ".format(
            key, param_grid[key][-1], u"\u00B1", np.std(param_grid[key]), time.asctime() ), end="\r")

        params3 = get_param_var(param_grid)
        val3 = param_grid[key][-1] + np.std(param_grid[key])
        params3[key] = val3 
        score3, params3 = compute_score(params3, param_var, param_grid, param_format, x_train, x_test, y_train, y_test)
        print("xxx {} {} {}{}  (--{}--)   ".format(
            key, param_grid[key][-1], u"\u00B1", np.std(param_grid[key]), time.asctime() ), end="\r")

        if score3 == min(score1, score2, score3): param_grid[key] = param_grid[key] + [params3[key]] 
        if score2 == min(score1, score2, score3): param_grid[key] = param_grid[key] + [params2[key]]
        if score1 == min(score1, score2, score3): param_grid[key] = param_grid[key] + [params1[key]]

        if max(score1, score2, score3) < best_score :
            best_score = min(score1, score2, score3)
            params = get_param_var(param_grid)
            best_grid = format_params(params, param_var, param_grid, param_format)
            max_iter = 100
        else:
            max_iter = max_iter - 1

        time_elapsed = time.time() - since
        print("{:3.0f}m {:2.0f}s {:.08f} {:.08f} {:>20} {:>13.08f} {}{:>13.08f}".format(
            time_elapsed // 60, time_elapsed % 60, max(score1, score2, score3), 
            best_score, key, param_grid[key][-1], u"\u00B1", np.std(param_grid[key]) ))

        if max_iter == 0 : break
        if start + 60*60*4 < time.time():break
        
    return best_grid

def get_best_grid(x_train, y_train, x_test, y_test):
    param_grid = {
        "learning_rate"    : [ 0.001, 0.3, 0.10] ,
        "n_estimators"     : [  100, 700, 100] ,
        "booster"          : ["gbtree"],
        "seed"             : [  42] ,
        "max_depth"        : [   2, 15, 5],
        "min_child_weight" : [   1, 20, 5],
        "max_delta_step"   : [   1, 20, 5],
        "scale_pos_weight" : [   1, 20, 5],
        "gamma"            : [ 0.0,  5.0, 0.5],
        "reg_alpha"        : [ 0.0,  5.0, 0.5],
        "reg_lambda"       : [ 0.0,  5.0, 0.5],
        "colsample_bylevel": [ 0.000001,  1.0, 0.7 ],
    #     "colsample_bynode" : [ 0.000001,  1.0, 1.0 ],
        "colsample_bytree" : [ 0.000001,  1.0, 0.7 ],
        "subsample"        : [ 0.000001,  1.0, 0.7 ],
        "objective"        : ["reg:squarederror"],
        "random_state"     : [ 42 ],
        "eval_metric"      : [ "mae" ],
    #     "use_label_encoder": [False],
    #     "base_score"       : [  0.000000000001, 1-0.000000000001, 0.5 ],
    }
    param_format = {
        "learning_rate"    : "{:.08f}" ,
        "n_estimators"     : "{:.0f}"  ,
        "seed"             : "{}"  ,
        "booster"          : "{}"  ,
        "max_depth"        : "{:.0f}"  ,
        "min_child_weight" : "{:.0f}"  ,
        "max_delta_step"   : "{:.0f}"  ,
        "scale_pos_weight" : "{:.0f}"  ,
        "gamma"            : "{:.08f}" , 
        "reg_alpha"        : "{:.08f}" , 
        "reg_lambda"       : "{:.08f}" , 
        "colsample_bylevel": "{:.08f}" ,
        "colsample_bynode" : "{:.08f}" ,
        "colsample_bytree" : "{:.08f}" ,
        "subsample"        : "{:.08f}" ,
        "objective"        : "{}"      ,
        "random_state"     : "{}"  ,
        "eval_metric"      : "{}"      ,
        "base_score"       : "{:.08f}" ,
    }
    
    print("MODEL A")
    param_grida = param_grid.copy()
    best_grida = find_best_param(param_grida, param_format, x_train, x_test, y_train[:,0], y_test[:,0])
    print("MODEL B")
    param_gridb = param_grid.copy()
    best_gridb = find_best_param(param_gridb, param_format, x_train, x_test, y_train[:,1], y_test[:,1])

    return best_grida, best_gridb

def bulid_models(x_train, y_train, x_test, y_test, best_grida, best_gridb):
    root_folder = lib.features.STORAGE
    file_patha = os.path.join(root_folder, "modela.xgb")
    file_pathb = os.path.join(root_folder, "modelb.xgb")
    
    modela = XGBRegressor()
    modela.load_model(file_patha)
    y_preda = modela.predict(x_test)
    base_scorea = mean_absolute_error(y_test[:,0], y_preda)

    modelb = XGBRegressor()
    modelb.load_model(file_pathb)
    y_predb = modela.predict(x_test)
    base_scoreb = mean_absolute_error(y_test[:,1], y_predb)

    modela = XGBRegressor(**best_grida)
    modela = modela.fit(x_train, y_train[:,0], eval_set=[(x_test, y_test[:,0])],
               early_stopping_rounds=100, verbose=False)    
    y_preda = modela.predict(x_test)
    scorea = mean_absolute_error(y_test[:,0], y_preda)
    print("score A : {} vs {}".format(scorea, base_scorea))
    if scorea <= base_scorea:
        modela.save_model(file_patha)
        print("model A saved !")
    
    modelb = XGBRegressor(**best_gridb)
    modelb = modelb.fit(x_train, y_train[:,1], eval_set=[(x_test, y_test[:,1])],
               early_stopping_rounds=100, verbose=False)
    y_predb = modelb.predict(x_test)
    scoreb = mean_absolute_error(y_test[:,1], y_predb)
    print("score B : {} vs {}".format(scoreb, base_scoreb))
    if scoreb <= base_scoreb:
        modelb.save_model(file_pathb)
        print("model B saved !")


if __name__ == "__main__":

    for _ in range(1000000):
        
        print(datetime.datetime.now())
        
        wait_time = 3 #random.randint(24, 72)
        if datetime.datetime.today().weekday() == 1:

            try:
                df = extract_data()
                lib.wallet.write_file("healthcheck", "OK")

    #             root_folder = lib.features.STORAGE
    #             file_path = os.path.join(root_folder, "extract.csv")
    #             df = pd.read_csv(file_path)
                x_train, y_train, x_test, y_test = get_data_set(df)
                best_grida, best_gridb = get_best_grid(x_train, y_train, x_test, y_test)

                bulid_models(x_train, y_train, x_test, y_test, best_grida, best_gridb)
                print(datetime.datetime.now())
                lib.wallet.write_file("healthcheck", "OK")
                wait_time = 24

            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("loop", exc_type, exc_traceback.tb_lineno, str(e))
                mro = exc_type.mro()
                error_lines = traceback.extract_tb(exc_traceback)
                if len(mro) > 0: 
                     print("Type: '{}'".format(mro[0].__name__))
                message = "Description: '{}'".format(exc_value)
                if len(error_lines) > 0:
                    message += " at line {}".format(error_lines[-1].lineno)
                for line in error_lines[::-1]:
                    print("Module: '{}' at line {}".format(line.name, line.lineno))
                  
        
        for minute in range(wait_time):
            lib.wallet.write_file("healthcheck", "OK")
            time.sleep(60*60)    
            
            