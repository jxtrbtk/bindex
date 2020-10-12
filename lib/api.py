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
