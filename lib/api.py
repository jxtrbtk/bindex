import requests
import json
import time
import os

import pandas as pd
import numpy as np
import random

import datetime

from . import config 

ROOT = config.ROOT
P_MARKET_PAGE_SIZE = 1000


def get_rj(res):
    time.sleep(1)
    r = requests.get(ROOT + res)
    return r.json()


def get_page(res, page=0, query=None, key=None):
    size = P_MARKET_PAGE_SIZE
    if page>0 : time.sleep(1)
    res = res + "?limit={}&offset={}".format(size, page*size)
    if query is not None:
        res = res + "&" + query
    rj = get_rj(res)
    if key is not None:
        rj = rj[key]
    if type(rj) == type({}):
        rj = []
    return rj


def get_all(res, key=None):
    data = []
    res_decomposition = res.split("?")
    query = None
    if len(res_decomposition) == 2:
        query = res_decomposition[1]
    res = res_decomposition[0]
    for p in range(1000):
        page = get_page(res, page=p, query=query, key=key)
        if len(page)>0:
            data += page
            if len(page)<P_MARKET_PAGE_SIZE: break
        else:
            break
    return data


def get_all_trades(symbol, d=3):
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

    return trades


def api_time(t=3):
    rj = get_rj("time")
    if ("ap_time" not in list(rj.keys())):
        if t > 0: 
            time.sleep(5)
            apt = api_time(t=t-1)
    else:
        apt = pd.to_datetime(rj["ap_time"])
    return apt


def broadcast(data):
    res = "broadcast?sync=1"
    headers = {'Accept': 'application/json','User-Agent': 'jxtr-bot'}
    headers["content-type"] = "text/plain"
    timeout = 10
    r = requests.post(ROOT + res, data=data, headers=headers)
    
    return r.json()    

