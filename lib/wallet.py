import os
import io
import time
import pandas as pd

from . import api
from . import dex

def read_file (filepath):
    content = ""
    with io.open(filepath, "r") as f: 
        content = f.read()
    return content

def write_file (filepath, line):
    with io.open(filepath, "r") as f: 
        f.write(str(line)+"\n")

def write_to_file (filepath, line):
    with io.open(filepath, "a") as f: 
        f.write(str(line)+"\n")

def get_secret_folder():
    folders = []
    folders.append(os.path.abspath(os.path.join(os.sep, "secret")))
    folders.append(os.path.join("secret"))
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

def get_balance(): 
    account = get_public_key()
    rj = api.get_rj("account/{}".format(account))
    balances = rj["balances"]

    return balances

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

