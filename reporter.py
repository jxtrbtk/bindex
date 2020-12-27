import os
import sys

import requests
import datetime
import time

import random

import lib
import lib.wallet

FOLDER = "secret"

def get_root_folder():
    folders = []
    folders.append(os.path.abspath(os.path.join(os.sep, FOLDER)))
    folders.append(os.path.join(FOLDER))
    for folder in folders:
        check_file = os.path.isdir(folder)
        if check_file: break
    else:
        folder = None
    return folder

# lib.wallet.get_balance(address="bnb1yhf94putzdlm5ymhqgr6x0rcru9jwvpe6ja6m2")

def get_wallets():
    root = get_root_folder()
    wallets = []
    for wallet in os.listdir(root):
        folder = os.path.join(root, wallet)
        if os.path.isdir(folder):
            filepath = os.path.join(folder, "wallet.pub.txt") 
            check_file = os.path.isfile(filepath)
            if check_file:
                address = lib.wallet.read_file(filepath)
                wallets.append(address)
    return wallets

def get_reporter_url():
    root = get_root_folder()
    url = ""
    filepath = os.path.join(root, "report_url.txt")
    check_file = os.path.isfile(filepath)
    if check_file:
        url = lib.wallet.read_file(filepath)
    
    filepath = os.path.join(root, "report_key.txt")
    check_file = os.path.isfile(filepath)
    if check_file:
        url = url + "?k={}".format(lib.wallet.read_file(filepath))
        
    return url

def get_ref_amount():
    url = get_reporter_url()
    url += "&ref=1"
    res = requests.get(url)
    return res.text

def report(wallet, amount):
    url = get_reporter_url()
    url += "&w={}&a={:.012f}".format(wallet, amount)
    res = requests.get(url)
    print(res.text)

def get_binance_ticker(symbol, key="weightedAvgPrice"):
    req = requests.get("https://api1.binance.com/api/v3/ticker/24hr?symbol={}".format(symbol))
    j = req.json()
    return float(j[key])

def report_all_wallets():
    wallets = get_wallets()
    wallets_content = { w:"" for w in wallets}
    for wallet in wallets:
        wallets_content[wallet] = lib.wallet.get_balance(address=wallet)
    # add reference ref AWC
    wallets.append("Ref1")
    wallets_content["Ref1"]=[{'free': '1.00000000', 'frozen': '0.00000000', 'locked': '0.00000000', 'symbol': 'AWC-986'}]
    value  = get_ref_amount()
    wallets.append("Ref")
    wallets_content["Ref"]=[{'free': str(value), 'frozen': '0.00000000', 'locked': '0.00000000', 'symbol': 'AWC-986'}]
    # add reference ref AWC 

    df = lib.market.get_markets()
    df["vwapPrice"]   = df["weightedAvgPrice"]
    df["refVolume"]   = df["volume24"]
    df["refQuote"]   = df["quoteVolume"]
    df = lib.market.calculate_base_bnb(df)

    ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    print("-- {} ---".format(ts))

    for wallet in wallets_content.keys():
        print(wallet, end=" ")
        balances = wallets_content[wallet]
        if len(balances) > 0:
            df_balance = lib.market.get_balance_bnb(df, balances)
            bnb = df_balance["total_BNB"].sum()
        else:
            bnb = 0.0
        print("{:.012f}".format(bnb), end=" ")
        report(wallet, bnb)

    for symbol in ["BNBEUR", "BNBBTC", "BTCEUR"]:
        print(symbol, get_binance_ticker(symbol))    
        report(symbol, get_binance_ticker(symbol))    


def main():
    for i in range(1000000):

        try:
            report_all_wallets()
            lib.wallet.write_file("healthcheck", "OK")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("operations error:", exc_type, exc_tb.tb_lineno, str(e))

        wait_time = 50 + random.randint(0, 7)    
        for minute in range(wait_time):
            print("{}/{}".format(minute, wait_time), end="\r")
            time.sleep(60)
    
if __name__ == "__main__":
    main()
