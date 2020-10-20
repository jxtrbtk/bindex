import requests
import json
import time
import os

import pandas as pd
import numpy as np
import random

import datetime

from . import config 
from . import api
from . import wallet

def get_markets():
    rj_ticker = api.get_all("ticker/24hr")
    rj_market = api.get_all("markets")
    dft = pd.DataFrame(rj_ticker, dtype=float)
    dfm = pd.DataFrame(rj_market, dtype=str) #we need lot_size and tick_size as str to keep in in Decimal
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
    hours_volume_ref = 1000 
    for idx in list(df.index):
        symbol = df.loc[idx, "pair"]
        if verbose: print("#K{:3}/{:3}-{}                 ".format(
            idx, len(list(df.index)), symbol), end="\r")
        dfk = get_klines(symbol=symbol)
        df.loc[idx, "refVolume"] = dfk["volume"].mean()
        df.loc[idx, "refQuote"]  = dfk["quote"].mean()
        df.loc[idx, "refCount"]  = dfk["count"].mean()
        if dfk["volume"].tail(hours_volume_ref).mean() > 0:
            df.loc[idx, "vwapPrice"] = dfk["quote"].tail(hours_volume_ref).mean() / dfk["volume"].tail(hours_volume_ref).mean()
        else:
            df.loc[idx, "vwapPrice"] = np.nan
    
    return df

def infuse_trades(df, verbose=False):
    df["statisticalSpread"]   = np.nan
    df["statisticalAskPrice"]   = np.nan
    df["statisticalBidPrice"]   = np.nan
    df["statisticalAskVolume"]   = np.nan
    df["statisticalBidVolume"]   = np.nan
    for idx in list(df.index):
        symbol = df.loc[idx, "pair"]
        if verbose: print("#T{:3}/{:3}-{}                 ".format(
            idx, len(list(df.index)), symbol), end="\r")
        tj = api.get_all_trades(symbol=symbol, d=3)
        dft = pd.DataFrame(tj, dtype="float")
        if len(dft) > 0:
            df_ask  = dft[dft["tickType"] == "BuyTaker"].copy()
            df_bid = dft[dft["tickType"] == "SellTaker"].copy()
            if df_ask.shape[0] > 0 and df_bid.shape[0] > 0:

                price = (dft["price"]*dft["quantity"]).sum() / dft["quantity"].sum()
                price_ask = (df_ask["price"]*df_ask["quantity"]).sum() / df_ask["quantity"].sum()
                price_bid = (df_bid["price"]*df_bid["quantity"]).sum() / df_bid["quantity"].sum()

                df.loc[idx, "statisticalSpread"]    = (price_ask - price_bid) / price
                df.loc[idx, "statisticalAskPrice"]  = price_ask
                df.loc[idx, "statisticalBidPrice"]  = price_bid
                df.loc[idx, "statisticalAskVolume"] = df_ask["quantity"].sum()
                df.loc[idx, "statisticalBidVolume"] = df_bid["quantity"].sum()
    
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

def calculate_score(df, df_balance=None): 
    df["score_count"] = df["refCount"]/df["refCount"].sum() 
    df["score_volume"] = df["volume_BNB"]/df["volume_BNB"].sum() 
    df["score_spread"] = df["statisticalSpread"]/df["statisticalSpread"].sum() 
    df["score"] = (df["score_count"]+df["score_volume"]+df["score_spread"])/3
    df = df.sort_values(by="score", ascending=False)
    df = df.reset_index(drop=True)
    df["cumsum"] = df["score"].cumsum()

    if df_balance is not None:
        total = df_balance["total_BNB"].sum()
        df["share_BNB"] = df["score"] * total
    
    return df

def pick_symbol(df):
    chance = random.random()
    df["test"] = (df["cumsum"] > chance)
    idx = df["test"].values.argmax()
    symbol = df["pair"].iloc[idx]

    return symbol


def get_balance_bnb(df): 
        
    balances = wallet.get_balance()
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
    df_balance["total_free_BNB"] = df_balance["free"] * df_balance["price"] 

    return df_balance
                

def qualify_from_balance(df, df_balance):

    total = df_balance["total_BNB"].sum()
    df["share_BNB"] = df["score"] * total
    df["thresholdAmount_BNB"] = df["lot_size"].astype(float) * df["priceBase_BNB"] * 24
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
    
    return df

def init_qualify(df):
    df["qualify"] = False
    return df
    
def get_qualified(df):
    return df[df["qualify"] == True].copy()

def get_orders():
    orders = wallet.get_orders()
    df_orders = pd.DataFrame(orders, dtype=float)
    return df_orders

def qualify_from_orders(df, df_orders):
    
    if df_orders.shape[0] > 0:
        pairs_available = list(df_orders["symbol"].unique())
        for p in pairs_available:
            where = (df["pair"] == p) 
            if where.sum() > 0:
                df.loc[where, "qualify"] = True    
    
    return df

def compute_invest_data(df, df_balance, df_orders):

    df["orderBuy"] = .0
    df["orderSell"] = .0
    for idx, order in df_orders.iterrows():
        pair = order["symbol"]
        side = order["side"]
        if side == 1: mode="Buy"
        if side == 2: mode="Sell"
        where = (df["pair"] == order["symbol"])
        df.loc[where, "order"+mode] += float(order["price"]) * float(order["quantity"])
    df["prodBuy"] = df["orderBuy"] * df["priceQuote_BNB"]
    df["prodSell"] = df["orderSell"] * df["priceQuote_BNB"]
    df["prod_BNB"] = df["prodSell"] + df["prodBuy"]

    df["shareBase"] = .0
    df["shareQuote"] = .0
    symbols_available = list(df["base_asset_symbol"])
    symbols_available += list(df["quote_asset_symbol"])
    symbols_available = list(set(symbols_available))
    for symbol in symbols_available:
        where1 = (df["base_asset_symbol"] == symbol)
        where2 = (df["quote_asset_symbol"] == symbol)
        total_score = df[where1 | where2]["score"].sum()
        df.loc[where1, "shareBase"] = df.loc[where1, "score"] / total_score
        df.loc[where2, "shareQuote"] = df.loc[where2, "score"] / total_score

    df["balanceBase"] = .0
    df["balanceQuote"] = .0
    for idx, token in df_balance.iterrows():
        symbol = token["symbol"]
        where1 = (df["base_asset_symbol"] == symbol)
        where2 = (df["quote_asset_symbol"] == symbol)
        df.loc[where1, "balanceBase"] = token["total_free_BNB"] * df.loc[where1, "shareBase"]
        df.loc[where2, "balanceQuote"] = token["total_free_BNB"] * df.loc[where2, "shareQuote"]

    df["operation_BNB"] = df[["balanceBase", "balanceQuote", "prodBuy", "prodSell"]].sum(axis=1)

    df["investQuote"] = df[["prodBuy", "prodSell"]].sum(axis=1) + df["balanceBase"] 
    df["investBase"] = df[["prodBuy", "prodSell"]].sum(axis=1) + df["balanceQuote"]
    df["investQuote"] = df["investQuote"] / df["share_BNB"]
    df["investBase"] = df["investBase"] / df["share_BNB"]
    
    return df 

def get_market_opportunities():
    #get simple market data
    df = get_markets()

    #refresh
    df = refresh_market_opportunities(df)

    return df

def refresh_tickers(df):
    rj_ticker = api.get_all("ticker/24hr")
    dft = pd.DataFrame(rj_ticker, dtype=float)
    dft = dft.rename(columns={"symbol":"pair", "volume":"volume24"})
    
    for idx, row in df.iterrows():
        pair = row["pair"]
        dft_tmp = dft[dft["pair"]==pair].reset_index()
        for col in dft_tmp.columns:
            df.loc[idx, col] = dft_tmp.loc[0, col] 
        
    return df


def refresh_market_opportunities(df):
    
    #refresh tickers values
    df = refresh_tickers(df)

    #get indicators from 1000h klines
    df = infuse_klines(df, verbose=True)
    #get indicators from trades (spread)
    df = infuse_trades(df, verbose=True)
    #compute reference values in BNB
    df = calculate_base_bnb(df)
    #calculate score on the whole dataset 
    df = calculate_score(df)

    #get token balance and current orders 
    df_balance = get_balance_bnb(df)
    df_orders = get_orders()

    #initialize "qualify" flag
    df = init_qualify(df)
    #qualify markets if enough budget given the min quantity and other settings
    #and markets were tokens are already hold
    df = qualify_from_balance(df, df_balance)
    #qualify markets where there is onging orders
    df = qualify_from_orders(df, df_orders)
    #filter only qualified markets
    df = get_qualified(df)

    #refine score on qualified only
    df = calculate_score(df, df_balance)
    #compute data to determine the portfolio balance 
    df = compute_invest_data(df, df_balance, df_orders)
    
    return df
