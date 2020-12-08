#!/usr/bin/env python
# coding: utf-8

import os
import sys
import datetime
import random
import json

import numpy as np
import pandas as pd

# import xgboost as xgb

import lib
import lib.finta as finta

STORAGE = os.path.join("collector","data")

def analyze_depth(df_asks, df_bids, data):
    data_out = {}
    where_b = (df_bids["price"] > (data["base_price_mid"]-data["base_price_std_weighted"]))
    df_bids[where_b] #.shape
    where_a = (df_asks["price"] < (data["base_price_mid"]+data["base_price_std_weighted"]))
    df_asks[where_a] #.shape
    # data["base_price_mid"]-data["base_price_std_weighted"], data["base_price_mid"]+data["base_price_std_weighted"]

    precision = 0.025
    for i in range(40):
        where_b = (df_bids["price"] > (data["base_price_mid"]-(i+1)*precision*data["base_price_std_weighted"]))
        where_a = (df_asks["price"] < (data["base_price_mid"]+(i+1)*precision*data["base_price_std_weighted"]))
        # data["base_price_mid"]-data["base_price_std_weighted"], data["base_price_mid"]+data["base_price_std_weighted"]
        d1 = df_bids.loc[where_b, "quantity"].sum() / (data["base_volume_sum"]/6) 
        d2 = df_asks.loc[where_a, "quantity"].sum() / (data["base_volume_sum"]/6)
        d3, d4 = df_bids.loc[where_b, "quantity"].sum() / data["base_volume_median"], df_asks.loc[where_a]["quantity"].sum() / data["base_volume_median"]
        d5, d6 = df_bids.loc[where_b, "quantity"].sum() / data["base_volume_average"], df_asks.loc[where_a]["quantity"].sum() / data["base_volume_average"]
        d7, d8 = df_bids[where_b].shape[0], df_asks[where_a].shape[0]
        data_out["depth_volume_ask_{:02d}".format(i+1)] = df_asks.loc[where_a, "quantity"].sum() #/ (data["base_volume_sum"]/6) 
        data_out["depth_volume_bid_{:02d}".format(i+1)] = df_bids.loc[where_b, "quantity"].sum() #/ (data["base_volume_sum"]/6) 
        data_out["depth_volume_ask_{:02d}_norm".format(i+1)] = data_out["depth_volume_ask_{:02d}".format(i+1)] / (data["base_volume_sum"]/6) 
        data_out["depth_volume_bid_{:02d}_norm".format(i+1)] = data_out["depth_volume_bid_{:02d}".format(i+1)] / (data["base_volume_sum"]/6) 
        data_out["depth_count_ask_{:02d}".format(i+1)] = df_asks[where_a].shape[0] 
        data_out["depth_count_bid_{:02d}".format(i+1)] = df_bids[where_b].shape[0]
        data_out["depth_count_ask_{:02d}_norm".format(i+1)] = df_asks[where_a].shape[0] /data["base_nb_trades"]
        data_out["depth_count_bid_{:02d}_norm".format(i+1)] = df_bids[where_b].shape[0] /data["base_nb_trades"]

    return data_out


def analyze_klines(df, prefix, price, volume, count):
    data_out = {}
    if prefix == "klines_1h":
        data_out["ADX"] = finta.TA.ADX(df).fillna(method="bfill").tail(1).mean()
        data_out["EV_MACD_SIGNAL"] = finta.TA.EV_MACD(df)["SIGNAL"].fillna(method="bfill").tail(1).mean()
        data_out["IFT_RSI"] = finta.TA.IFT_RSI(df).fillna(method="bfill").tail(13).mean()
        data_out["WOBV"] = finta.TA.WOBV(df).fillna(method="bfill").tail(1).mean()
        data_out["STC"] = finta.TA.STC(df).fillna(method="bfill").tail(13).mean()
        data_out["BBWIDTH"] = finta.TA.BBWIDTH(df).fillna(method="bfill").tail(8).mean()
        data_out["EFI"] = finta.TA.EFI(df).fillna(method="bfill").tail(13).mean() 
        data_out["VZO"] = finta.TA.VZO(df).fillna(method="bfill").tail(13).mean()
        data_out["STOCHRSI"] = finta.TA.STOCHRSI(df).fillna(method="bfill").tail(1).mean()
        data_out["AO"] = finta.TA.AO(df).fillna(method="bfill").tail(13).mean()
        data_out["PZO"] = finta.TA.PZO(df).fillna(method="bfill").tail(13).mean()
        data_out["VORTEX"] = finta.TA.VORTEX(df).fillna(method="bfill").tail(8).mean()
        data_out["PPO_HISTO"] = finta.TA.PPO(df)["HISTO"].fillna(method="bfill").tail(1).mean()
        data_out["ATR"] = finta.TA.ATR(df).fillna(method="bfill").tail(13).mean()
        data_out["QSTICK"] = finta.TA.QSTICK(df).fillna(method="bfill").tail(8).mean()
        data_out["TSI"] = finta.TA.TSI(df).fillna(method="bfill").tail(13).mean()
        data_out["EV_MACD"] = finta.TA.EV_MACD(df)["MACD"].fillna(method="bfill").tail(1).mean()
        data_out["MSD"] = finta.TA.MSD(df).fillna(method="bfill").tail(8).mean()
        data_out["PPO_PPO"] = finta.TA.PPO(df)["PPO"].fillna(method="bfill").tail(8).mean()
        data_out["TRIX"] = finta.TA.TRIX(df).fillna(method="bfill").tail(1).mean()
        data_out["CMO"] = finta.TA.CMO(df).fillna(method="bfill").tail(13).mean()
        data_out["RSI"] = finta.TA.RSI(df).fillna(method="bfill").tail(1).mean()
        data_out["PPO_SIGNAL"] = finta.TA.PPO(df)["SIGNAL"].fillna(method="bfill").tail(13).mean()
        data_out["FISH"] = finta.TA.FISH(df).fillna(method="bfill").tail(8).mean()
        data_out["CCI"] = finta.TA.CCI(df).fillna(method="bfill").tail(1).mean()
        data_out["COPP"] = finta.TA.COPP(df).fillna(method="bfill").tail(13).mean()
        data_out["MOM"] = finta.TA.MOM(df).fillna(method="bfill").tail(13).mean()

    if prefix == "klines_5m":
        data_out["VFI"] = finta.TA.VFI(df).fillna(method="bfill").tail(1).mean()
        data_out["ADX"] = finta.TA.ADX(df).fillna(method="bfill").tail(8).mean() 
        data_out["FISH"] = finta.TA.FISH(df).fillna(method="bfill").tail(1).mean()
        data_out["WOBV"] = finta.TA.WOBV(df).fillna(method="bfill").tail(1).mean()
        data_out["CFI"] = finta.TA.CFI(df).fillna(method="bfill").tail(1).mean()
        data_out["EFI"] = finta.TA.EFI(df).fillna(method="bfill").tail(1).mean()
        data_out["AO"] = finta.TA.AO(df).fillna(method="bfill").tail(1).mean()
        data_out["IFT_RSI"] = finta.TA.IFT_RSI(df).fillna(method="bfill").tail(1).mean()
        data_out["PPO_HISTO"] = finta.TA.PPO(df)["HISTO"].fillna(method="bfill").tail(13).mean() 
        data_out["VZO"] = finta.TA.VZO(df).fillna(method="bfill").tail(8).mean()
        data_out["BBWIDTH"] = finta.TA.BBWIDTH(df).fillna(method="bfill").tail(8).mean()
        data_out["RSI"] = finta.TA.RSI(df).fillna(method="bfill").tail(8).mean() 
        data_out["TRIX"] = finta.TA.TRIX(df).fillna(method="bfill").tail(1).mean()
        data_out["MOM"] = finta.TA.MOM(df).fillna(method="bfill").tail(1).mean()
        data_out["PZO"] = finta.TA.PZO(df).fillna(method="bfill").tail(8).mean() 
        data_out["MSD"] = finta.TA.MSD(df).fillna(method="bfill").tail(13).mean() 
        data_out["CMO"] = finta.TA.CMO(df).fillna(method="bfill").tail(13).mean() 
        data_out["PPO_SIGNAL"] = finta.TA.PPO(df)["SIGNAL"].fillna(method="bfill").tail(1).mean() 
        data_out["TR"] = finta.TA.TR(df).fillna(method="bfill").tail(8).mean() 
        data_out["PPO"] = finta.TA.PPO(df).fillna(method="bfill").tail(8).mean() 
        data_out["EV_MACD"] = finta.TA.EV_MACD(df)["SIGNAL"].fillna(method="bfill").tail(8).mean()
        data_out["TSI"] = finta.TA.TSI(df)["TSI"].fillna(method="bfill").tail(8).mean() 
        data_out["QSTICK"] = finta.TA.QSTICK(df).fillna(method="bfill").tail(8).mean() 

        
    if prefix == "klines_1h":
        data_out["P_BBANDS_BB_LOWER"] = finta.TA.BBANDS(df)["BB_UPPER"].fillna(method="bfill").tail(8).mean()
        data_out["P_BBANDS_BB_MIDDLE"] = finta.TA.BBANDS(df)["BB_MIDDLE"].fillna(method="bfill").tail(8).mean()
        data_out["P_BBANDS_BB_LOWER"] = finta.TA.BBANDS(df)["BB_LOWER"].fillna(method="bfill").tail(8).mean()
        data_out["P_DO_LOWER"] = finta.TA.DO(df)["UPPER"].fillna(method="bfill").tail(8).mean()
        data_out["P_DO_MIDDLE"] = finta.TA.DO(df)["MIDDLE"].fillna(method="bfill").tail(8).mean()
        data_out["P_DO_LOWER"] = finta.TA.DO(df)["LOWER"].fillna(method="bfill").tail(8).mean()
        data_out["P_PIVOT_s4"] = finta.TA.PIVOT(df)["s4"].fillna(method="bfill").tail(8).mean()
        data_out["P_PIVOT_r4"] = finta.TA.PIVOT(df)["r4"].fillna(method="bfill").tail(8).mean()
        data_out["P_PIVOT_s2"] = finta.TA.PIVOT(df)["s2"].fillna(method="bfill").tail(13).mean()
        data_out["P_PIVOT_r2"] = finta.TA.PIVOT(df)["r2"].fillna(method="bfill").tail(13).mean()
        data_out["P_PIVOT_FIB_s2"] = finta.TA.PIVOT_FIB(df)["s2"].fillna(method="bfill").tail(13).mean()
        data_out["P_PIVOT_FIB_r2"] = finta.TA.PIVOT_FIB(df)["r2"].fillna(method="bfill").tail(13).mean()
        data_out["P_SAR"] = finta.TA.SAR(df).fillna(method="bfill").tail(1).mean()
        data_out["P_ICHIMOKU_senkou_span_a"] =  finta.TA.ICHIMOKU(df)["senkou_span_a"].fillna(method="bfill").tail(8).mean()
        data_out["P_EVWMA"] =  finta.TA.EVWMA(df, 34).fillna(method="bfill").tail(13).mean()
        data_out["P_DEMA"] =  finta.TA.DEMA(df, 34).fillna(method="bfill").tail(1).mean()
        data_out["P_EMA"] =  finta.TA.EMA(df, 13).fillna(method="bfill").tail(8).mean()
       
    if prefix == "klines_5m":        
        data_out["P_DMI_DI+"] = finta.TA.DMI(df)["DI+"].fillna(method="bfill").tail(1).mean()
        data_out["P_DMI_DI-"] = finta.TA.DMI(df)["DI-"].fillna(method="bfill").tail(8).mean()
        data_out["P_SAR"] = finta.TA.SAR(df).fillna(method="bfill").tail(8).mean()
        data_out["P_EVWMA"] = finta.TA.EVWMA(df, 34).fillna(method="bfill").tail(13).mean()
        data_out["P_APZ_LOWER"] = finta.TA.APZ(df)["LOWER"].fillna(method="bfill").tail(13).mean()
        data_out["P_CHANDELIER_Long"] = finta.TA.CHANDELIER(df)["Long."].fillna(method="bfill").tail(1).mean()
        data_out["P_ICHIMOKU_CHIKOU"] = finta.TA.ICHIMOKU(df)["CHIKOU"].fillna(method="bfill").tail(1).mean()
        data_out["P_SSMA"] = finta.TA.SSMA(df, 5).fillna(method="bfill").tail(1).mean()
        data_out["P_ZLEMA"] = finta.TA.ZLEMA(df, 34).fillna(method="bfill").tail(1).mean()
        data_out["P_TRIMA"] = finta.TA.TRIMA(df, 21).fillna(method="bfill").tail(13).mean()
        data_out["P_EVWMA"] = finta.TA.EVWMA(df, 5).fillna(method="bfill").tail(1).mean()

    if prefix == "klines_1h":
        data_out["volume_ewm"] = df["volume"].ewm(ignore_na=False, span=34, adjust=True).mean().fillna(method="bfill").tail(13).mean()
        data_out["count_ewm"] = df["count"].ewm(ignore_na=False, span=13, adjust=True).mean().fillna(method="bfill").tail(13).mean()
        
    if prefix == "klines_5m":        
        data_out["volume_ewm"] = df["volume"].ewm(ignore_na=False, span=13, adjust=True).mean().fillna(method="bfill").tail(1).mean()
        data_out["count_ewm"] = df["count"].ewm(ignore_na=False, span=34, adjust=True).mean().fillna(method="bfill").tail(1).mean()
        
    # new !
    data_out["P_open"] = df["open"].fillna(method="bfill").tail(1).mean()
    data_out["P_high"] = df["high"].fillna(method="bfill").tail(1).mean()
    data_out["P_low"] = df["low"].fillna(method="bfill").tail(1).mean()
    data_out["P_close"] = df["close"].fillna(method="bfill").tail(1).mean()
    data_out["P_vwap"] = (df["volume"] / df["quote"]).fillna(method="bfill").tail(1).mean()
    data_out["Open_rank"] = df["open"].fillna(method="bfill").rank(pct=True).tail(1).mean()
    data_out["High_rank"] = df["high"].fillna(method="bfill").rank(pct=True).tail(1).mean()
    data_out["Low_rank"] = df["low"].fillna(method="bfill").rank(pct=True).tail(1).mean()
    data_out["Close_rank"] = df["close"].fillna(method="bfill").rank(pct=True).tail(1).mean()
    
    for p in [34, 55, 89, 144, 233, 377, 610]:
        data_out["Open_rank_"+str(p)] = df["open"].fillna(method="bfill").tail(p).rank(pct=True).tail(1).mean()
        data_out["High_rank_"+str(p)] = df["high"].fillna(method="bfill").tail(p).rank(pct=True).tail(1).mean()
        data_out["Low_rank_"+str(p)] = df["low"].fillna(method="bfill").tail(p).rank(pct=True).tail(1).mean()
        data_out["Close_rank_"+str(p)] = df["close"].fillna(method="bfill").tail(p).rank(pct=True).tail(1).mean()
    # new !

    for k in data_out:
        k_mod = k.replace(prefix + "_", "")
        if k_mod[:2] == "P_":
            data_out[k] = data_out[k] / price
        if k_mod[:7] == "volume_" or k_mod[:6] == "quote_":
            data_out[k] = data_out[k] / volume
        if k_mod[:6] == "count_" :
            data_out[k] = data_out[k] / count 
    
    data_out = {prefix+"_"+e:v for e,v in data_out.items()}

    return data_out


def analyze_ticker(df_ticker, price):
    data = {}
    data["base_symbol"] = df_ticker["symbol"].values[0] 
    data["base_ticker_weightedAvgPrice"] = df_ticker["weightedAvgPrice"].values[0] 
    data["ticker_askPrice_weightedAvgPrice"] = (df_ticker["askPrice"] / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_askPrice_lastPrice"] = (df_ticker["askPrice"] / df_ticker["lastPrice"]).values[0]
    data["ticker_askPrice_data_price_std"] = (df_ticker["askPrice"] / price).values[0]
    data["ticker_bidPrice_weightedAvgPrice"] = (df_ticker["bidPrice"] / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_bidPrice_lastPrice"] = (df_ticker["bidPrice"] / df_ticker["lastPrice"]).values[0]
    data["ticker_bidPrice_base_price_mid"] = (df_ticker["bidPrice"] / price).values[0]
    data["ticker_spread_weightedAvgPrice"] = ((df_ticker["askPrice"]-df_ticker["bidPrice"]) / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_spread_lastPrice"] = ((df_ticker["askPrice"]-df_ticker["bidPrice"]) / df_ticker["lastPrice"]).values[0]
    data["ticker_spread_lastPrice"] = ((df_ticker["askPrice"]-df_ticker["bidPrice"]) / price).values[0]
    data["ticker_spread_diff_sum"] = 2*((df_ticker["askPrice"]-df_ticker["bidPrice"]) / (df_ticker["askPrice"]+df_ticker["bidPrice"])).values[0]
    data["ticker_count"] = df_ticker["count"].values[0]
    data["ticker_highPrice_weightedAvgPrice"] = (df_ticker["highPrice"] / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_highPrice_lastPrice"] = (df_ticker["highPrice"] / df_ticker["lastPrice"]).values[0]
    data["ticker_highPrice_base_price_mid"] = (df_ticker["highPrice"] / price).values[0]
    data["ticker_lowPrice_weightedAvgPrice"] = (df_ticker["lowPrice"] / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_lowPrice_lastPrice"] = (df_ticker["lowPrice"] / df_ticker["lastPrice"]).values[0]
    data["ticker_lowPrice_base_price_mid"] = (df_ticker["lowPrice"] /price).values[0]
    data["ticker_spreadHighLow_weightedAvgPrice"] = ((df_ticker["highPrice"]-df_ticker["lowPrice"]) / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_spreadHighLow_lastPrice"] = ((df_ticker["highPrice"]-df_ticker["lowPrice"]) / df_ticker["lastPrice"]).values[0]
    data["ticker_spreadHighLow_base_price_mid"] = ((df_ticker["highPrice"]-df_ticker["lowPrice"]) / price).values[0]
    data["ticker_spreadHighLow_diff_sum"] = 2*((df_ticker["highPrice"]-df_ticker["lowPrice"]) / (df_ticker["askPrice"]+df_ticker["bidPrice"])).values[0]
    data["ticker_priceChange_weightedAvgPrice"] = (df_ticker["priceChange"] / df_ticker["weightedAvgPrice"]).values[0]
    data["ticker_priceChange_lastPrice"] = (df_ticker["priceChange"] / df_ticker["lastPrice"]).values[0]
    data["ticker_priceChange_base_price_mid"] = (df_ticker["priceChange"] / price).values[0]
    data["ticker_priceChangePercent"] = df_ticker["priceChangePercent"].values[0]
    data["ticker_askQuantity_totalQuantity"] = df_ticker["askQuantity"].values[0] / (df_ticker["askQuantity"].values[0] + df_ticker["bidQuantity"].values[0])
    data["ticker_bidQuantity_totalQuantity"] = df_ticker["bidQuantity"].values[0] / (df_ticker["askQuantity"].values[0] + df_ticker["bidQuantity"].values[0])
    data["ticker_askQuantity_lastQuantity"] = df_ticker["askQuantity"].values[0] / df_ticker["lastQuantity"].values[0]
    data["ticker_bidQuantity_lastQuantity"] = df_ticker["bidQuantity"].values[0] / df_ticker["lastQuantity"].values[0]
    
    return data


def analyze_tickers(df_tickers, symbol):
    data = {}
    df_tickers["refVolume"] = df_tickers["volume"]
    df_tickers["refQuote"]  = df_tickers["quoteVolume"]
    df_tickers["refCount"]  = df_tickers["count"]
    df_tickers["vwapPrice"] = df_tickers["weightedAvgPrice"]
    df_tickers = lib.market.calculate_base_bnb(df_tickers)
    mask = (df_tickers["symbol"] == symbol)
    data["tickers_volume_BNB"] = df_tickers[mask]["volume_BNB"].values[0]
    data["tickers_volume_BNB_rank"] = df_tickers["volume_BNB"].rank(pct=True)[mask].values[0]
    data["tickers_volume_BNB_normal"] = ((df_tickers["volume_BNB"][mask]-df_tickers["volume_BNB"].mean())/df_tickers["volume_BNB"].std()).values[0]
    data["tickers_volume_BNB_lognormal"] = ((np.log(df_tickers["volume_BNB"].replace(0, np.nan))[mask]-np.log(df_tickers["volume_BNB"].replace(0, np.nan)).mean())/np.log(df_tickers["volume_BNB"].replace(0, np.nan)).std()).values[0]
    data["tickers_count_rank"] = df_tickers["count"].rank(pct=True)[mask].values[0]
    data["tickers_count_normal"] = ((df_tickers["count"][mask]-df_tickers["count"].mean())/df_tickers["count"].std()).values[0]
    data["tickers_count_lognormal"] = ((np.log(df_tickers["count"].replace(0, np.nan))[mask]-np.log(df_tickers["count"].replace(0, np.nan)).mean())/np.log(df_tickers["count"].replace(0, np.nan)).std()).values[0]
    data["tickers_priceChangePercent_rank"] = df_tickers["priceChangePercent"].rank(pct=True)[mask].values[0]
    data["tickers_priceChangePercent_normal"] = ((df_tickers["priceChangePercent"][mask]-df_tickers["priceChangePercent"].mean())/df_tickers["priceChangePercent"].std()).values[0]
    data["tickers_priceChangePercent_lognormal"] = ((np.log(100+df_tickers["priceChangePercent"].replace(0, np.nan))[mask]-np.log(100+df_tickers["priceChangePercent"].replace(0, np.nan)).mean())/np.log(100+df_tickers["priceChangePercent"].replace(0, np.nan)).std()).values[0]
    data["tickers_priceChangePercent_priceChangeWeightedMean"] = (df_tickers[mask]["priceChangePercent"] / ( (df_tickers["priceChangePercent"] * df_tickers["volume_BNB"]).sum() / df_tickers["volume_BNB"].sum())).values[0]
    return data


def init_base_data(df_trades, df_asks, df_bids, ap_time):
    data = {}
    data["base_ap_time"] =  ap_time.timestamp()
    data["base_volume_median"] = df_trades["quantity"].median()
    data["base_volume_average"] = df_trades["quantity"].mean()
    data["base_volume_sum"] = df_trades["quantity"].sum()
    data["base_nb_trades"] = df_trades.shape[0]

    df_asks["q_sum"] = df_asks["quantity"].cumsum()
    df_bids["q_sum"] = df_bids["quantity"].cumsum()
    mask = df_asks["q_sum"]>=data["base_volume_median"] 
    price_a = df_asks.loc[mask, "price"].min()
    mask = df_bids["q_sum"]>=data["base_volume_median"] 
    price_b = df_bids.loc[mask, "price"].max()
    data["base_price_mid"] = (price_a + price_b) / 2

    price_avg = (df_trades["price"] * df_trades["quantity"]).sum() / df_trades["quantity"].sum() 
    price_var = np.average((df_trades["price"]-price_avg)**2, weights=df_trades["quantity"])
    data_price_std = np.sqrt(price_var)
    if data_price_std == 0.0: raise ValueError("data_price_std cannot be equal to zero (0.0)")
    data["base_price_std_weighted"] = data_price_std
    data["base_price_std"] = df_trades["price"].std()
    
    return data


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def build_source_dataset(folder):
    j_time = load_json(os.path.join(folder, "time.json"))
    j_depth = load_json(os.path.join(folder, "depth.json"))
    j_klines_1h = load_json(os.path.join(folder, "klines_1h.json"))
    j_klines_5m = load_json(os.path.join(folder, "klines_5m.json"))
    j_ticker = load_json(os.path.join(folder, "ticker.json"))
    j_tickers = load_json(os.path.join(folder, "tickers.json"))
    j_trades = load_json(os.path.join(folder, "trades_3d.json"))
    j_data = (j_time, j_depth, j_klines_1h, j_klines_5m, j_ticker, j_tickers, j_trades)
    return j_data  

def build_destination_dataset(folder):
    j_time = load_json(os.path.join(folder, "time.json"))
    j_klines_5m_delayed_3d = load_json(os.path.join(folder, "klines_5m_delayed_3d.json"))
    j_data = (j_time, j_klines_5m_delayed_3d)
    return j_data

def build_x(j_data):

    j_time, j_depth, j_klines_1h, j_klines_5m, j_ticker, j_tickers, j_trades = j_data

    ap_time = pd.to_datetime(j_time["ap_time"])

    cols = ["price", "quantity"]
    df_asks = pd.DataFrame(j_depth["asks"], columns=cols, dtype=float)
    df_bids = pd.DataFrame(j_depth["bids"], columns=cols, dtype=float)

    cols = ["time", "open", "high", "low", "close", "volume", "end", "quote", "count"]
    df_klines_1h = pd.DataFrame(j_klines_1h, columns=cols, dtype=float)
    df_klines_5m = pd.DataFrame(j_klines_5m, columns=cols, dtype=float)
    df_ticker  = pd.DataFrame(j_ticker, dtype=float)
    df_tickers = pd.DataFrame(j_tickers, dtype=float)
    df_trades = pd.DataFrame(j_trades, dtype="float")
    if df_trades.shape[0] == 0: raise ValueError("trades dataframe empty")

    data = init_base_data(df_trades, df_asks, df_bids, ap_time)
    data.update(analyze_ticker(df_ticker, data["base_price_mid"]))
    data.update(analyze_tickers(df_tickers, data["base_symbol"]))
    data.update(analyze_depth(df_asks, df_bids, data))
    data.update(analyze_klines(df_klines_5m, "klines_5m",
                    data["base_price_mid"], data["base_volume_sum"], data["base_nb_trades"]))
    data.update(analyze_klines(df_klines_1h, "klines_1h",
                    data["base_price_mid"], data["base_volume_sum"], data["base_nb_trades"]))
    data.update(analyze_trades(df_trades, j_time,
                    data["base_price_mid"], data["base_volume_sum"], data["base_nb_trades"]))

    return data


def get_future_high_low(j_klines_5m_delayed_3d, j_time):
    cols = ["time", "open", "high", "low", "close", "volume", "end", "quote", "count"]
    df_j_klines_5m_delayed_3d = pd.DataFrame(j_klines_5m_delayed_3d, columns=cols, dtype=float)
    df_j_klines_5m_delayed_3d["time"] = (df_j_klines_5m_delayed_3d["time"] / 1000).astype(int)
    
    ap_time = pd.to_datetime(j_time["ap_time"])
    t0 =  ap_time.timestamp()
    t1 =  t0 + 60 * 60 * 24 * 1

    where = (df_j_klines_5m_delayed_3d["time"]>t0)
    where = where & (df_j_klines_5m_delayed_3d["time"]<t1)

    high = df_j_klines_5m_delayed_3d[where]["high"].max()
    low = df_j_klines_5m_delayed_3d[where]["low"].min()
    
    return high, low


def analyze_trades(df, j_time, price, volume, count):
    mask_buy = (df["tickType"]=="BuyTaker")
    mask_sell = (df["tickType"]=="SellTaker")
    df = df[["time", "tradeId", "tickType", "price", "quantity"]].copy()
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["buy_price"] = np.nan
    df["buy_quantity"] = np.nan
    df["sell_price"] = np.nan
    df["sell_quantity"] = np.nan
    df.loc[mask_buy, "buy_price"] = df.loc[mask_buy, "price"]
    df.loc[mask_buy, "buy_quantity"] = df.loc[mask_buy, "quantity"]
    df.loc[mask_sell, "sell_price"] = df.loc[mask_sell, "price"]
    df.loc[mask_sell, "sell_quantity"] = df.loc[mask_sell, "quantity"]

    data = {}
    gold = 0.618
    # for p in [gold**0, gold, gold**2, gold**3, gold**4, gold**5, gold**6]:
    for i in range(7):
        p = gold**i
        mask = (df["time"] > (pd.to_datetime(j_time["ap_time"]) - datetime.timedelta(days=p*3)))
        data["count_sell_"+str(i)] = df[mask & mask_sell]["tradeId"].count() / count
        data["count_buy_"+str(i)] = df[mask & mask_buy]["tradeId"].count() / count
        data["count_total_"+str(i)] = df[mask]["tradeId"].count() / count
        data["quantity_sell_"+str(i)] = df[mask & mask_sell]["quantity"].sum() / volume
        data["quantity_buy_"+str(i)] = df[mask & mask_buy]["quantity"].sum() / volume
        data["quantity_total_"+str(i)] = df[mask]["quantity"].sum() / volume
        if df[mask]["quantity"].fillna(1).sum() == 0.0:
            if i > 0: data["P_vwap_total_"+str(i)] = data["P_vwap_total_"+str(i-1)]
            else: data["P_vwap_total_"+str(i)] = price
        else:
            data["P_vwap_total_"+str(i)] = ( ((df[mask]["quantity"]*df[mask]["price"])).fillna(0).sum() / df[mask]["quantity"].fillna(1).sum())
        if df[mask & mask_sell]["quantity"].fillna(1).sum() == 0.0:
            if i > 0: data["P_vwap_sell_"+str(i)] = data["P_vwap_sell_"+str(i-1)]
            else: data["P_vwap_sell_"+str(i)] = data["P_vwap_total_"+str(i)]
        else:
            data["P_vwap_sell_"+str(i)] = (( (df[mask & mask_sell]["quantity"]*df[mask & mask_sell]["price"])).fillna(0).sum() / df[mask & mask_sell]["quantity"].fillna(1).sum())
        if df[mask & mask_buy]["quantity"].fillna(1).sum() == 0.0:
            if i > 0: data["P_vwap_buy_"+str(i)] = data["P_vwap_buy_"+str(i-1)]
            else: data["P_vwap_buy_"+str(i)] = data["P_vwap_total_"+str(i)]
        else:
            data["P_vwap_buy_"+str(i)] = (( (df[mask & mask_buy]["quantity"]*df[mask & mask_buy]["price"])).fillna(0).sum() / df[mask & mask_buy]["quantity"].fillna(1).sum())
        data["P_vwap_sell_"+str(i)] = data["P_vwap_sell_"+str(i)] / price 
        data["P_vwap_buy_"+str(i)] = data["P_vwap_buy_"+str(i)] / price
        data["P_vwap_total_"+str(i)] = data["P_vwap_total_"+str(i)] / price

        data["spread_"+str(i)] = np.nan_to_num(2 * (data["P_vwap_buy_"+str(i)] - data["P_vwap_sell_"+str(i)]) / data["P_vwap_total_"+str(i)])
        data["frequency_"+str(i)] = 2 * (data["count_buy_"+str(i)] - data["count_sell_"+str(i)]) / data["count_total_"+str(i)]
        data["pressure_"+str(i)] = 2 * (data["quantity_buy_"+str(i)] - data["quantity_sell_"+str(i)]) / data["quantity_total_"+str(i)]
    # df
    data = {"trades_"+e:v for e,v in data.items()}
    
    return data



if __name__ == "__main__":
    root_folder = STORAGE
    i,ie = 0,0
    data = []
    for symbol in os.listdir(root_folder):
        symbol_folder = os.path.join(root_folder,symbol)
        for ts in os.listdir(symbol_folder):
            ts_folder = os.path.join(symbol_folder,ts)

            try:
                if os.path.exists(os.path.join(ts_folder, "klines_5m_delayed_3d.json")):
                    j_data = build_source_dataset(ts_folder)
                    data_x = build_x(j_data)

                    j_data = build_destination_dataset(ts_folder)
                    j_klines_5m_delayed_3d = j_data[-1]
                    j_time = j_data[0]
                    h,l = get_future_high_low(j_klines_5m_delayed_3d, j_time)        
                    data_x["target_high"] = h
                    data_x["target_low"] = l

                    data.append(data_x)

                    i = i + 1

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("operations error:", exc_type, exc_tb.tb_lineno, str(e))
                ie = ie + 1

            print(i, ie, ts_folder)

    df = pd.DataFrame(dtype=float).from_dict(data)
    df.to_csv("extract.csv", index=False)
