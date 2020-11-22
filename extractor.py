#!/usr/bin/env python
# coding: utf-8

import os
import random
import json

import numpy as np
import pandas as pd

import xgboost as xgb

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
    df_bids.loc[where_b, "quantity"].sum() / (data["base_volume_sum"]/6) #, 
    df_asks.loc[where_a]["quantity"].sum() / (data["base_volume_sum"]/6)
    df_bids.loc[where_b, "quantity"].sum() / data["base_volume_median"], df_asks.loc[where_a]["quantity"].sum() / data["base_volume_median"]
    df_bids.loc[where_b, "quantity"].sum() / data["base_volume_average"], df_asks.loc[where_a]["quantity"].sum() / data["base_volume_average"]
    df_bids[where_b].shape[0], df_asks[where_a].shape[0]

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
        data_out["depth_count_ask_{:02d}".format(i)] = df_asks[where_a].shape[0] 
        data_out["depth_count_bid_{:02d}".format(i)] = df_bids[where_b].shape[0]
        data_out["depth_count_ask_{:02d}_norm".format(i)] = df_asks[where_a].shape[0] /data["base_nb_trades"]
        data_out["depth_count_bid_{:02d}_norm".format(i)] = df_bids[where_b].shape[0] /data["base_nb_trades"]

    return data_out


def analyze_klines(df, prefix, price, volume, count):
    df["MACD"] = finta.TA.MACD(df)["MACD"]
    df["MACD_signal"] = finta.TA.MACD(df)["SIGNAL"]
    df["PPO"] = finta.TA.PPO(df)["PPO"]
    df["PPO_signal"] = finta.TA.PPO(df)["SIGNAL"]
    df["PPO_histo"] = finta.TA.PPO(df)["HISTO"]
    df["AO"] = finta.TA.AO(df)
    df["RSI"] = finta.TA.RSI(df)
    df["STOCH"] = finta.TA.STOCH(df)
    df["STOCHD"] = finta.TA.STOCHD(df)
    df["VORTEX"] = finta.TA.VORTEX(df)["VIm"]
    df["VORTEX"] = finta.TA.VORTEX(df)["VIp"]
    df["CHAIKIN"] = finta.TA.CHAIKIN(df)
    df["TRIX"] = finta.TA.TRIX(df)
    df["ER"] = finta.TA.ER(df)
    df["MOM"] = finta.TA.MOM(df)
    df["ROC"] = finta.TA.ROC(df)
    df["IFT_RSI"] = finta.TA.IFT_RSI(df)
    df["TR"] = finta.TA.TR(df)
    df["ATR"] = finta.TA.ATR(df)
    df["P_SAR"] = finta.TA.SAR(df)
    df["BBWIDTH"] = finta.TA.BBWIDTH(df)
    df["PERCENT_B"] = finta.TA.PERCENT_B(df)
    df["ADX"] = finta.TA.ADX(df)
    df["STOCHRSI"] = finta.TA.STOCHRSI(df)
    df["WILLIAMS"] = finta.TA.WILLIAMS(df)
    df["MI"] = finta.TA.MI(df)
    df["P_TP"] = finta.TA.TP(df)
    df["ADL"] = finta.TA.ADL(df)
    df["MFI"] = finta.TA.MFI(df)
    df["OBV"] = finta.TA.OBV(df)
    df["WOBV"] = finta.TA.WOBV(df)
    df["VZO"] = finta.TA.VZO(df)
    df["PZO"] = finta.TA.PZO(df)
    df["EFI"] = finta.TA.EFI(df)
    df["CFI"] = finta.TA.CFI(df)
    df["EMV"] = finta.TA.EMV(df)
    df["CCI"] = finta.TA.CCI(df)
    df["COPP"] = finta.TA.COPP(df)
    df["CMO"] = finta.TA.CMO(df)
    df["QSTICK"] = finta.TA.QSTICK(df)
    df["FISH"] = finta.TA.FISH(df)
    df["SQZMI"] = finta.TA.SQZMI(df)
    df["VPT"] = finta.TA.VPT(df)
    df["FVE"] = finta.TA.FVE(df)
    df["VFI"] = finta.TA.VFI(df)
    df["MSD"] = finta.TA.MSD(df)
    df["STC"] = finta.TA.STC(df)

    df["EV_MACD_MACD"] = finta.TA.EV_MACD(df)["MACD"]
    df["EV_MACD_SIGNAL"] = finta.TA.EV_MACD(df)["SIGNAL"]
    df["P_BBANDS_BB_UPPER"] = finta.TA.BBANDS(df)["BB_UPPER"]
    df["P_BBANDS_BB_MIDDLE"] = finta.TA.BBANDS(df)["BB_MIDDLE"]
    df["P_BBANDS_BB_LOWER"] = finta.TA.BBANDS(df)["BB_LOWER"]
    df["P_MOBO_BB_UPPER"] = finta.TA.MOBO(df)["BB_UPPER"]
    df["P_MOBO_BB_MIDDLE"] = finta.TA.MOBO(df)["BB_MIDDLE"]
    df["P_MOBO_BB_LOWER"] = finta.TA.MOBO(df)["BB_LOWER"]
    df["P_KC_KC_UPPER"] = finta.TA.KC(df)["KC_UPPER"]
    df["P_KC_KC_LOWER"] = finta.TA.KC(df)["KC_LOWER"]
    df["P_DO_LOWER"] = finta.TA.DO(df)["LOWER"]
    df["P_DO_MIDDLE"] = finta.TA.DO(df)["MIDDLE"]
    df["P_DO_UPPER"] = finta.TA.DO(df)["UPPER"]
    df["P_DMI_DI+"] = finta.TA.DMI(df)["DI+"] 
    df["P_DMI_DI-"] = finta.TA.DMI(df)["DI-"]
    df["P_PIVOT_pivot"] = finta.TA.PIVOT(df)["pivot"]
    df["P_PIVOT_s1"] = finta.TA.PIVOT(df)["s1"]
    df["P_PIVOT_s2"] = finta.TA.PIVOT(df)["s2"]
    df["P_PIVOT_s3"] = finta.TA.PIVOT(df)["s3"]
    df["P_PIVOT_s4"] = finta.TA.PIVOT(df)["s4"]
    df["P_PIVOT_r1"] = finta.TA.PIVOT(df)["r1"]
    df["P_PIVOT_r2"] = finta.TA.PIVOT(df)["r2"]
    df["P_PIVOT_r3"] = finta.TA.PIVOT(df)["r3"]
    df["P_PIVOT_r4"] = finta.TA.PIVOT(df)["r4"]
    df["P_PIVOT_FIB_pivot"] = finta.TA.PIVOT(df)["pivot"]
    df["P_PIVOT_FIB_s1"] = finta.TA.PIVOT_FIB(df)["s1"]
    df["P_PIVOT_FIB_s2"] = finta.TA.PIVOT_FIB(df)["s2"]
    df["P_PIVOT_FIB_s3"] = finta.TA.PIVOT_FIB(df)["s3"]
    df["P_PIVOT_FIB_s4"] = finta.TA.PIVOT_FIB(df)["s4"]
    df["P_PIVOT_FIB_r1"] = finta.TA.PIVOT_FIB(df)["r1"]
    df["P_PIVOT_FIB_r2"] = finta.TA.PIVOT_FIB(df)["r2"]
    df["P_PIVOT_FIB_r3"] = finta.TA.PIVOT_FIB(df)["r3"]
    df["P_PIVOT_FIB_r4"] = finta.TA.PIVOT_FIB(df)["r4"]
    df["KST_KST"] = finta.TA.KST(df)["KST"]
    df["KST_signal"] = finta.TA.KST(df)["signal"]
    df["TSI_TSI"] = finta.TA.TSI(df)["TSI"]
    df["TSI_signal"] = finta.TA.TSI(df)["signal"]
    df["EBBP_Bull"] = finta.TA.EBBP(df)["Bull."]
    df["EBBP_Bear"] = finta.TA.EBBP(df)["Bear."]
    df["BASP_Buy"] = finta.TA.BASP(df)["Buy."]
    df["BASP_Sell"] = finta.TA.BASP(df)["Sell."]
    df["BASPN_Buy"] = finta.TA.BASPN(df)["Buy."]
    df["BASPN_Sell"] = finta.TA.BASPN(df)["Sell."]
    df["P_CHANDELIER_Short"] = finta.TA.CHANDELIER(df)["Short."]
    df["P_CHANDELIER_Long"] = finta.TA.CHANDELIER(df)["Long."]
    df["WTO_WT1"] = finta.TA.WTO(df)["WT1."]
    df["WTO_WT2"] = finta.TA.WTO(df)["WT2."]
    df["P_ICHIMOKU_TENKAN"] = finta.TA.ICHIMOKU(df)["TENKAN"]
    df["P_ICHIMOKU_KIJUN"] = finta.TA.ICHIMOKU(df)["KIJUN"]
    df["P_ICHIMOKU_senkou_span_a"] = finta.TA.ICHIMOKU(df)["senkou_span_a"]
    df["P_ICHIMOKU_SENKOU"] = finta.TA.ICHIMOKU(df)["SENKOU"]
    df["P_ICHIMOKU_CHIKOU"] = finta.TA.ICHIMOKU(df)["CHIKOU"]
    df["P_APZ_UPPER"] = finta.TA.APZ(df)["UPPER"]
    df["P_APZ_LOWER"] = finta.TA.APZ(df)["LOWER"]

    for p in [2,3,5,8,13,21,34]:
        df["P_SMA_"+str(p)]   = finta.TA.SMA(df, p)
        df["P_SMM_"+str(p)]   = finta.TA.SMM(df, p)
        df["P_SSMA_"+str(p)]  = finta.TA.SSMA(df, p)
        df["P_EMA_"+str(p)]   = finta.TA.EMA(df, p)
        df["P_DEMA_"+str(p)]  = finta.TA.DEMA(df, p)
        df["P_TEMA_"+str(p)]  = finta.TA.TEMA(df, p)
        df["P_TRIMA_"+str(p)] = finta.TA.TRIMA(df, p)
        df["P_VAMA_"+str(p)]  = finta.TA.VAMA(df, p)
        df["P_KAMA_"+str(p)]  = finta.TA.KAMA(df, p)
        df["P_ZLEMA_"+str(p)] = finta.TA.ZLEMA(df, p)
        df["P_WMA_"+str(p)]   = finta.TA.WMA(df, p)
        df["P_HMA_"+str(p)]   = finta.TA.HMA(df, p)
        df["P_EVWMA_"+str(p)] = finta.TA.EVWMA(df, p) 
        df["P_SMMA_"+str(p)]  = finta.TA.SMMA(df, p)
        p_even = int(p/2)*2
        df["P_FRAMA_"+str(p)] = finta.TA.FRAMA(df, p_even)    
        df["volume_ewm_"+str(p)] = df["volume"].ewm(ignore_na=False, span=p, adjust=True).mean()
        df["quote_ewm_"+str(p)] = df["quote"].ewm(ignore_na=False, span=p, adjust=True).mean()
        df["count_ewm_"+str(p)] = df["count"].ewm(ignore_na=False, span=p, adjust=True).mean()

    df = df.fillna(method="bfill")

    d = df.tail(1).transpose().to_dict()
    k = list(d.keys())[-1]
    df_dict = d[k]

    data_out = {prefix + "_" + k : v for k, v in df_dict.items()}

    for p in [8,13]:
        for c in df.columns:
            key = prefix + "_" + c + "_avg" + str(p)
            val = df[c].tail(p).mean()
            data_out[key] = val

    for k in data_out:
        k_mod = k.replace(prefix + "_", "")
        if k_mod[:2] == "P_":
            data_out[k] = data_out[k] / price
        if k_mod[:7] == "volume_" or k_mod[:6] == "quote_":
            data_out[k] = data_out[k] / volume
        if k_mod[:6] == "count_" :
            data_out[k] = data_out[k] / count 
            
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


if __name__ == "__main__":
    root_folder = STORAGE
    i,e = 0,0
    data = []
    for symbol in os.listdir(root_folder):
        symbol_folder = os.path.join(root_folder,symbol)
        for ts in os.listdir(symbol_folder):
            ts_folder = os.path.join(symbol_folder,ts)

            try: 
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

            except:
                e = e + 1

            print(i, e, ts_folder)

    df = pd.DataFrame(dtype=float).from_dict(data)
    df.to_csv("extract.csv", index=False)




