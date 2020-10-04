import requests
import json
import time
import os

import pandas as pd
import numpy as np
import random

import sys
import traceback
import datetime

from . import config 

ROOT = config.ROOT


def get_rj(res):
    time.sleep(1)
    r = requests.get(ROOT + res)
    return r.json()

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

