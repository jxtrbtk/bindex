import os
import io

import time
import datetime

import pandas as pd

from . import api
from . import dex

FOLDER = "wallet"

def read_file (filepath):
    content = ""
    with io.open(filepath, "r") as f: 
        content = f.read()
    return content

def write_file (filepath, line):
    with io.open(filepath, "w") as f: 
        f.write(str(line)+"\n")

def write_to_file (filepath, line):
    with io.open(filepath, "a") as f: 
        f.write(str(line)+"\n")

def get_secret_folder():
    folders = []
    folders.append(os.path.abspath(os.path.join(os.sep, FOLDER)))
    folders.append(os.path.join(FOLDER))
    for folder in folders:
        check_file = os.path.isfile(os.path.join(folder, "wallet.pub.txt"))
        check_file = check_file & os.path.isfile(os.path.join(folder, "wallet.pk.txt"))
        if check_file: break
    else:
        folder = None 
    
    return folder

def get_public_key():
    secret_folder = get_secret_folder()
    filepath = os.path.join(secret_folder, "wallet.pub.txt")
    return read_file(filepath)

def get_private_key():    
    secret_folder = get_secret_folder()
    filepath = os.path.join(secret_folder, "wallet.pk.txt")
    return read_file(filepath)

def get_balance(address=None): 
    if address is None: address = get_public_key()
    rj = api.get_rj("account/{}".format(address))
    if "balances" in rj.keys():
        balances = rj["balances"]
    else:
        balances = []

    return balances

def get_trades_for_period(start, end, account, symbol=None):
    res = "trades?limit=1000&start={}&end={}&address={}".format(int(start), int(end), account) 
    if symbol is not None: 
        res += "&symbol={}".format(symbol) 
    rj = api.get_rj(res)

    return rj["trade"]

def get_trades(account=None, symbol=None):
    if account is None: account = get_public_key()
    trades = []
    start0 = 1595633691955 
    end = datetime.datetime.timestamp(api.api_time())*1000
    for nj in range(10000000):
        rj = get_trades_for_period(start0, end, account, symbol)
        if len(rj)>0:
            start = min([e["time"] for e in trades + rj])+1
            rj = get_trades_for_period(start, end, account, symbol)
            if len(rj)>0:
                trades = trades + rj
                end = start
            else:
                break
        if start == start0: break
    
    return  trades

def get_orders(symbol=None):
    account = get_public_key()
    res = "orders/open?address={}".format(account)
    if symbol is not None:
        res += "&symbol={}".format(symbol) 
    orders = api.get_all(res, key="order")

    return orders

def send_order(mode, quantity, price, symbol):
    wallet_address = get_public_key()
    wallet_private_key = get_private_key()
    time.sleep(1)
    res= dex.send_order(wallet_address, wallet_private_key, mode, quantity, price, symbol)    

    return res


def cancel_order(symbol, order_id):
    wallet_address = get_public_key()
    wallet_private_key = get_private_key()
    time.sleep(1)
    res= dex.cancel_order(wallet_address, wallet_private_key, symbol, order_id)    

    return res

