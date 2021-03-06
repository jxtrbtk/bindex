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
    rj = api.get_r(res).json()
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

    mask1 = (df["quoteAssetName"] == "BNB")
    df.loc[mask1, "volume_BNB"] = df["refQuote"]
    df.loc[mask1, "priceBase_BNB"] = df["weightedAvgPrice"]
    df.loc[mask1, "priceQuote_BNB"] = 1
    df.loc[mask1, "type"] = "STANDARD"

    mask2 = (df["baseAssetName"] == "BNB")
    df.loc[mask2, "volume_BNB"] = df["refVolume"]
    df.loc[mask2, "priceBase_BNB"] = 1
    df.loc[mask2, "priceQuote_BNB"] = 1/df["weightedAvgPrice"]
    df.loc[mask2, "type"] = "REVERSE"

    # AVA-645_BUSD-BD1 
    # AVA-645_BNB
    mask3 = (df["volume_BNB"].isna())
    for index, row in df[mask3].iterrows():
        mask = (df["baseAssetName"] == row["baseAssetName"])
        mask = mask & (df["quoteAssetName"] == "BNB")
        if df[mask].shape[0] == 1:
            df.loc[index, "volume_BNB"] = row["refVolume"] * df[mask].iloc[0]["vwapPrice"]
            df.loc[index, "priceBase_BNB"] = df[mask].iloc[0]["weightedAvgPrice"]
            df.loc[index, "priceQuote_BNB"] = df[mask].iloc[0]["weightedAvgPrice"] / row["weightedAvgPrice"]
            df.loc[index, "type"] = "BASEJOIN"


    # BTCB-1DE_BUSD-BD1
    #      BNB_BUSD-BD1
    mask4 = (df["volume_BNB"].isna())
    for index, row in df[mask4].iterrows():
        mask = (df["quoteAssetName"] == row["quoteAssetName"])
        mask = mask & (df["baseAssetName"] == "BNB")
        if df[mask].shape[0] == 1:
            df.loc[index, "volume_BNB"] = row["refQuote"] / df[mask].iloc[0]["vwapPrice"]
            df.loc[index, "priceBase_BNB"] = row["weightedAvgPrice"] / df[mask].iloc[0]["weightedAvgPrice"]
            df.loc[index, "priceQuote_BNB"] = 1 / df[mask].iloc[0]["weightedAvgPrice"]
            df.loc[index, "type"] = "QUOTEJOIN"

    mask3 = mask3 & ~mask4

    return df

def calculate_past_margin(df): 
    df["past_margin"] = 0.0

    trades = wallet.get_trades()
    address = wallet.get_public_key()
    symbols = []
    if len(trades) > 0: 
        df_trades = pd.DataFrame(trades, dtype=float)
        df_trades["amount"] = df_trades["price"]*df_trades["quantity"]
        symbols = list(df_trades["symbol"].unique())

    for symbol in symbols:
        found = 0
        mask = (df_trades["symbol"]==symbol)
        mask = mask & (df_trades["buyerId"]==address)
        if df_trades[mask]["quantity"].sum() > 0:
            price_buy = df_trades[mask]["amount"].sum() / df_trades[mask]["quantity"].sum()
            found += 1
        mask = (df_trades["symbol"]==symbol)
        mask = mask & (df_trades["sellerId"]==address)
        if df_trades[mask]["quantity"].sum() > 0:
            price_sell = df_trades[mask]["amount"].sum() / df_trades[mask]["quantity"].sum()
            found += 1
        mask = (df["pair"]==symbol)
        if found == 2:
            df.loc[mask, "past_margin"] = 2*(price_sell-price_buy)/(price_buy+price_sell)
        else:
            df.loc[mask, "past_margin"] = 0.0

    return df


def calculate_score(df, df_balance=None): 
    if "past_margin" not in df.columns:
        df["past_margin"] = 0.0
    
    df["score_count"]  = 0.0
    df["score_volume"] = 0.0 
    df["score_spread"] = 0.0 
    df["score_margin"] = 0.0 
    df["score"]        = 0.0
    df["cumsum"]       = 0.0
    
    mask = (df["qualify"] == True)
    df.loc[mask, "score_count"]  = df.loc[mask, "refCount"].fillna(0)/df.loc[mask, "refCount"].sum() 
    df.loc[mask, "score_volume"] = df.loc[mask, "volume_BNB"].fillna(0)/df.loc[mask, "volume_BNB"].sum() 
    df.loc[mask, "score_spread"] = df.loc[mask, "statisticalSpread"].fillna(0)/df.loc[mask, "statisticalSpread"].sum() 
    mask0 = (df["score_spread"]<0.0)
    df.loc[mask0, "score_spread"] = 0.0
    df.loc[mask, "score_margin"] = df["past_margin"] - df["past_margin"].min() 
    
    df.loc[mask, "score_count"]  = df.loc[mask, "score_count"]  / df.loc[mask, "score_count"].sum()
    df.loc[mask, "score_volume"] = df.loc[mask, "score_volume"] / df.loc[mask, "score_volume"].sum() 
    df.loc[mask, "score_spread"] = df.loc[mask, "score_spread"] / df.loc[mask, "score_spread"].sum()
    if df.loc[mask, "score_margin"].sum() <= 0:
        df.loc[mask, "score_margin"] = 1.0 
    df.loc[mask, "score_margin"] = df.loc[mask, "score_margin"] / df.loc[mask, "score_margin"].sum() 
    
    df.loc[mask, "score"] = (df.loc[mask, "score_count"]+df.loc[mask, "score_volume"]+ \
                             df.loc[mask, "score_spread"]+df.loc[mask, "score_margin"])/4
    
    mask0 = (df["score"]<=0.0)
    df.loc[mask0, "score"] = 0.0
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


def get_balance_bnb(df, balances=None): 
        
    if balances is None:
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
            if mask_s.sum() > 0:
                idx_s = df[mask_s].index[0]
                df.loc[idx_s, "qualify"] = True
    
    return df

def init_qualify(df, value=False):
    df["qualify"] = value
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
    #get indicators from trades history (past gain)
    df = calculate_past_margin(df)
    
    #compute reference values in BNB
    df = calculate_base_bnb(df)
    #calculate score on the whole dataset 
    df = init_qualify(df, True)
    df = calculate_score(df)

    #get token balance and current orders 
    df_balance = get_balance_bnb(df)
    df_orders = get_orders()

    #initialize "qualify" flag
    df = init_qualify(df, False)
    #qualify markets if enough budget given the min quantity and other settings
    #and markets were tokens are already hold
    df = qualify_from_balance(df, df_balance)
    #qualify markets where there is onging orders
    df = qualify_from_orders(df, df_orders)
    #refine score
    df = calculate_score(df, df_balance)

    #include dependancies
    df = qualify_dependancies(df)
    #keep only qualified (including dependancies at score 0.0)
    df = get_qualified(df)
    #compute data to determine the portfolio balance 
    df = compute_invest_data(df, df_balance, df_orders)
    
    return df


def qualify_dependancies(df):

    # search symbols for all BASEJOIN
    mask = (df["qualify"]==True)
    mask = mask & (df["type"] == "BASEJOIN") 
    symbols = list(df[mask]["base_asset_symbol"])

    #qualify converter to BNB
    where = df["base_asset_symbol"].isin(symbols)
    where = where & (df["quote_asset_symbol"] == "BNB")
    df.loc[where, "qualify"] = True

    # search symbols for all QUOTEJOIN
    mask = (df["qualify"]==True)
    mask = mask & (df["type"] == "QUOTEJOIN") 
    symbols = list(df[mask]["quote_asset_symbol"])
    
    #qualify converter to BNB
    where = df["quote_asset_symbol"].isin(symbols)
    where = where & (df["base_asset_symbol"] == "BNB")
    df.loc[where, "qualify"] = True
    
    df = df.sort_values(by=["qualify", "score"], ascending=False)
    df = df.reset_index(drop=True)

    return df
    