import requests
import json
import time
import os

import pandas as pd
import random

import sys
import traceback
import datetime

ROOT = "https://dex-european.binance.org/api/v1/"
STORAGE = "data"

## sudo docker build -t collector:v0.1 docker/collector/. 
## sudo docker run -it -v collector:/data -v secret:/secret collector:v0.1

def get_rj(res):
    time.sleep(2)
    r = requests.get(ROOT + res)
    return r.json()

def collect_data(symbol, folder, data_map=None):
    if data_map is None: data_map = get_data_map(symbol)
    for file, res in data_map:
        rj = get_rj(res)
        path = os.path.join(folder, file)
        with open(path, 'w') as j:
            json.dump(rj, j)

def get_data_map(symbol):
    data_map = [
        ["time.json",        "time"],
        ["klines_1h.json",   "klines?limit=1000&symbol={}&interval=1h".format(symbol)],    
        ["klines_5m.json",   "klines?limit=1000&symbol={}&interval=5m".format(symbol)],    
        ["ticker.json",      "ticker/24hr?symbol={}".format(symbol)],
#         ["trades.json",      "trades?symbol={}&limit=1000".format(symbol)],
        ["depth.json",       "depth?symbol={}&limit=1000".format(symbol)]
    ]
    return data_map            
            
P_MARKET_PAGE_SIZE = 1000
def get_page(res, page=0):
    size = P_MARKET_PAGE_SIZE
    if page>0 : time.sleep(2)
    res = res + "?limit={}&offset={}".format(size, page*size)
    rj = get_rj(res)
    if type(rj) == type({}):
        rj = []
    return rj

def get_all(res):
    markets = []
    for p in range(1000):
        data = get_page(res, page=p)
        if len(data)>0:
            markets += data
            if len(data)<P_MARKET_PAGE_SIZE: break
        else:
            break
    return markets

def get_all_markets(folder):
    rj = get_all("ticker/24hr")
    file = "tickers.json"
    path = os.path.join(folder, file)
    with open(path, 'w') as j:
        json.dump(rj, j)

def get_symbol():
    dfm = pd.DataFrame(get_all("ticker/24hr"))
    dfm = dfm[dfm["quoteAssetName"]=="BNB"]
    vol_total = dfm["quoteVolume"].astype("float").sum()
    dfm["ratio"] = dfm["quoteVolume"].astype("float") / vol_total
    dfm = dfm[["baseAssetName", "quoteAssetName", "ratio"]]
    dfm = dfm.sort_values(by="ratio", ascending=False)
    dfm["cumsum"] = dfm["ratio"].cumsum()
    chance = random.random()
    dfm["test"] = (dfm["cumsum"] < chance)
    idx = dfm["test"].values.argmin()
    symbol = "_".join(list(dfm[["baseAssetName","quoteAssetName"]].iloc[idx]))
    return symbol

def complete_data_collection():
    delta = 60*60*24*2 
    jfile = "klines_5m_delayed_3d.json"
    for symbol in os.listdir(STORAGE):
        sfolder = os.path.join(STORAGE, symbol)
        if os.path.isdir(sfolder):
            for ts in os.listdir(sfolder):
                tfolder = os.path.join(sfolder, ts)
                res = "klines?limit=1000&symbol={}&interval=5m".format(symbol)
                if os.path.isdir(tfolder) and ts.isnumeric():
                    ts0 = int(ts)
                    datafile_path = os.path.join(sfolder, jfile)
                    if not os.path.exists(datafile_path):
                        if (ts0+delta) < int(time.time()):
                            collect_data(symbol, tfolder, data_map=[[jfile, res]])

def create_folders(symbol):
                                        
    if not os.path.exists(STORAGE): 
        os.mkdir(STORAGE)

    symbol_folder = os.path.join(STORAGE, symbol)
    if not os.path.exists(symbol_folder): 
        os.mkdir(symbol_folder)

    ts = str(int(time.time()))
    data_folder = os.path.join(symbol_folder, ts)
    if not os.path.exists(data_folder): 
        os.mkdir(data_folder)
    
    return data_folder

def api_time(t=3):
    rj = get_rj("time")
    if ("ap_time" not in list(rj.keys())):
        if t > 0: 
            time.sleep(5)
            apt = api_time(t=t-1)
    else:
        apt = pd.to_datetime(rj["ap_time"])
    return apt

def get_all_trades(symbol, folder, d=3):
    call_limit = 1000
    trades = []
    end0 = end = datetime.datetime.timestamp(api_time())*1000
    start0 = start = end - d*24*60*60*1000  
    for nj in range(1000):
        res = "trades?symbol={}&limit={}&start={}&end={}".format(symbol, call_limit, int(start0), int(end))
        rj1 = get_rj(res)
        if len(rj1["trade"])==call_limit:
            # second call is a bit paranoid: 
            # prevent from 2 items at the exact same time 
            # and at the cut limit, tests shows such things happens !!
            start = min([e["time"] for e in trades + rj1["trade"]])+1
            res = "trades?symbol={}&limit={}&start={}&end={}".format(symbol, call_limit, int(start), int(end))
            rj2 = get_rj(res)
            if len(rj2["trade"])>0:
                trades = trades + rj2["trade"]
            end = start
        elif len(rj1["trade"])>0:
            trades = trades + rj1["trade"]
            break
        else: break

    file = "trades_{}d.json".format(d)
    path = os.path.join(folder, file)
    with open(path, 'w') as j:
        json.dump(trades, j)


if __name__ == "__main__":

    for _ in range(1000000):
        wait_time = random.randint(1, 120)
        try:
            symbol=get_symbol()
            print("{:>24} {:>20} > {}m".format(symbol, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), wait_time))
            data_folder = create_folders(symbol)
            collect_data(symbol, data_folder)
            get_all_markets(data_folder)
            get_all_trades(symbol, data_folder)
            complete_data_collection()

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
            time.sleep(60)
