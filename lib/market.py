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

ROOT = "https://dex-european.binance.org/api/v1/"
STORAGE = "data"

from . import config 
from . import api


def get_markets():
    rj_ticker = api.get_all("ticker/24hr")
    rj_market = api.get_all("markets")
    dft = pd.DataFrame(rj_ticker, dtype=float)
    dfm = pd.DataFrame(rj_market, dtype=float)
    dfm["symbol"] = dfm["base_asset_symbol"]+"_"+dfm["quote_asset_symbol"]
    
    df = dfm.merge(dft)
    df = df.rename(columns={"symbol":"pair", "volume":"volume24"})
    
    return df

def get_klines(symbol="AWC-986_BNB", interval="1h"):
    time.sleep(2/10)
    res  = "klines?limit={}&symbol={}&interval={}".format(
        1000, symbol, interval) #500 ~20j, 200~8j 
    rj = api.get_rj(res)
    cols = ["time", "open", "high", "low", "close", "volume", "end", "quote", "count"]
    df = pd.DataFrame(rj, columns=cols, dtype=float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True) 
    df["end"] = pd.to_datetime(df["end"], unit="ms", utc=True) 
    df["vwap"] = df["quote"] / df["volume"]  

    return df


def infuse_klines(df, verbose=False):
    df["vwapPrice"]   = np.nan
    df["refVolume"]   = np.nan
    df["refQuote"]   = np.nan
    df["refCount"]   = np.nan
    hours_volume_ref = 500 
    for idx in list(df.index):
        symbol = df.loc[idx, "pair"]
        if verbose: print("#{:3}/{:3}-{}".format(idx, len(list(df.index)), symbol), end="\r")
        dfk = get_klines(symbol=symbol)
        df.loc[idx, "refVolume"] = dfk["volume"].mean()
        df.loc[idx, "refQuote"]  = dfk["quote"].mean()
        df.loc[idx, "refCount"]  = dfk["count"].mean()
        if dfk["volume"].tail(hours_volume_ref).mean() > 0:
            df.loc[idx, "vwapPrice"] = dfk["quote"].tail(hours_volume_ref).mean() / dfk["volume"].tail(hours_volume_ref).mean()
        else:
            df.loc[idx, "vwapPrice"] = np.nan
    
    return df


def calculate_base_bnb(df):
    df["volume_BNB"]     = None
    df["priceBase_BNB"]  = None
    df["priceQuote_BNB"] = None
    df["type"] = None

    mask1 = (df["quote_asset_symbol"] == "BNB")
    df.loc[mask1, "volume_BNB"] = df["refQuote"]
    df.loc[mask1, "priceBase_BNB"] = df["weightedAvgPrice"]
    df.loc[mask1, "priceQuote_BNB"] = 1
    df.loc[mask1, "type"] = "STANDARD"

    mask2 = (df["base_asset_symbol"] == "BNB")
    df.loc[mask2, "volume_BNB"] = df["refVolume"]
    df.loc[mask2, "priceBase_BNB"] = 1
    df.loc[mask2, "priceQuote_BNB"] = 1/df["weightedAvgPrice"]
    df.loc[mask2, "type"] = "REVERSE"

    # AVA-645_BUSD-BD1 
    # AVA-645_BNB
    mask3 = (df["volume_BNB"].isna())
    for index, row in df[mask3].iterrows():
        mask = (df["base_asset_symbol"] == row["base_asset_symbol"])
        mask = mask & (df["quote_asset_symbol"] == "BNB")
        if df[mask].shape[0] == 1:
            df.loc[index, "volume_BNB"] = row["refVolume"] * df[mask].iloc[0]["vwapPrice"]
            df.loc[index, "priceBase_BNB"] = df[mask].iloc[0]["weightedAvgPrice"]
            df.loc[index, "priceQuote_BNB"] = df[mask].iloc[0]["weightedAvgPrice"] / row["weightedAvgPrice"]
            df.loc[index, "type"] = "BASEJOIN"


    # BTCB-1DE_BUSD-BD1
    #      BNB_BUSD-BD1
    mask4 = (df["volume_BNB"].isna())
    for index, row in df[mask3].iterrows():
        mask = (df["quote_asset_symbol"] == row["quote_asset_symbol"])
        mask = mask & (df["base_asset_symbol"] == "BNB")
        if df[mask].shape[0] == 1:
            df.loc[index, "volume_BNB"] = row["refQuote"] / df[mask].iloc[0]["vwapPrice"]
            df.loc[index, "priceBase_BNB"] = row["weightedAvgPrice"] / df[mask].iloc[0]["weightedAvgPrice"]
            df.loc[index, "priceQuote_BNB"] = 1 / df[mask].iloc[0]["weightedAvgPrice"]
            df.loc[index, "type"] = "QUOTEJOIN"

    mask3 = mask3 & ~mask4

    return df

def calculate_score(df): 
    df["score_count"] = df["refCount"]/df["refCount"].sum() 
    df["score_volume"] = df["volume_BNB"]/df["volume_BNB"].sum() 
    df["score"] = (df["score_count"]+df["score_volume"])/2
    df = df.sort_values(by="score", ascending=False)
    df = df.reset_index(drop=True)
    df["cumsum"] = df["score"].cumsum()

    return df

def pick_symbol(df):
    chance = random.random()
    df["test"] = (df["cumsum"] > chance)
    idx = df["test"].values.argmax()
    symbol = df["pair"].iloc[idx]

    return symbol


def get_balance_bnb(df, account): 
    rj = api.get_rj("account/{}".format(account))
    balances = rj["balances"]

    df_balance = pd.DataFrame(balances, dtype=float)
    df_balance["total"] = df_balance["free"]+df_balance["locked"]
    df_balance["price"] = None
    for e in balances:
        symbol = e["symbol"]
        price = 0
        if symbol == "BNB":
            price = 1
        else:
            test = df["pair"].str.contains(symbol)
            mask = (df["type"] == "STANDARD")
            if (test & mask).sum() >= 1:
                price = float(df.loc[(test & mask), "priceBase_BNB"].head(1))
            mask = (df["type"] == "REVERSE")
            if (test & mask).sum() >= 1:
                price = float(df.loc[(test & mask), "priceQuote_BNB"].head(1))
            mask3 = (df["type"] == "BASEJOIN")
            mask4 = (df["type"] == "QUOTEJOIN")
        df_balance.loc[df_balance["symbol"] == symbol, "price"] = price
            
    df_balance["total_BNB"] = df_balance["total"] * df_balance["price"] 

    return df_balance
                
                
def qualify_from_balance(df, df_balance):

    df["qualify"] = False 

    total = df_balance["total_BNB"].sum()
    df["share_BNB"] = df["score"] * total
    df["thresholdAmount_BNB"] = df["lot_size"] * df["priceBase_BNB"] * 24
    mask = (df["share_BNB"] > df["thresholdAmount_BNB"])
    df.loc[mask, "qualify"] = True

    symbols_available = list(df_balance["symbol"])
    mask = df["qualify"]
    for s in symbols_available:
        test1 = not (s in df[mask]["base_asset_symbol"].values)
        test2 = not (s in df[mask]["quote_asset_symbol"].values)
        if (test1 & test2):
            mask_s = df["pair"].str.contains(s)
            idx_s = df[mask_s].index[0]
            df.loc[idx_s, "qualify"] = True
    
    return df[df["qualify"]].copy()                