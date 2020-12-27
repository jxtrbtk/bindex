import sys

import numpy as np
import pandas as pd

import time
import datetime
import random
from decimal import Decimal


import lib

RATIO = 1/3
INVEST_TRG_HIGH = 0.99
INVEST_TRG_LOW = 0.01

# sudo docker run -it -v secret:/secret_data --name operatorQN --restart unless-stopped operator:vQN.1

def choose_price(t_data):
    symbol = t_data["pair"]

    # get trades on 3 days
    tj = lib.api.get_all_trades(symbol, d=3)

    dft = pd.DataFrame(tj, dtype="float")

    # calculate weighted standard deviation
    price_avg = (dft["price"] * dft["quantity"]).sum() / dft["quantity"].sum() 
    price_var = np.average((dft["price"]-price_avg)**2, weights=dft["quantity"])
    data_price_std = np.sqrt(price_var)
    
    # get order book
    baj = lib.api.get_rj("depth?symbol={}&limit=1000".format(symbol))

    cols = ["price", "quantity"]
    dfa = pd.DataFrame(baj["asks"], columns=cols, dtype=float)
    dfb = pd.DataFrame(baj["bids"], columns=cols, dtype=float)

    #filter entries under the median, find price ask & bid
    vol_med = dft["quantity"].median()
    dfa["q_sum"] = dfa["quantity"].cumsum()
    dfb["q_sum"] = dfb["quantity"].cumsum()
    mask = dfa["q_sum"]>=vol_med 
    price_a = dfa.loc[mask, "price"].min()
    mask = dfb["q_sum"]>=vol_med 
    price_b = dfb.loc[mask, "price"].max()

    #middle price
    data_price_mid = (price_a + price_b) / 2
    
    return data_price_mid, data_price_std

def init_budget(t_data):
    orders = lib.wallet.get_orders(symbol=t_data["pair"])
    df_orders = pd.DataFrame(orders, dtype=float)
    budget = 0.0
    if df_orders.shape[0] > 0:
        budget_in_quote = (df_orders["quantity"]*df_orders["price"]).sum()
        budget = budget_in_quote * t_data["priceQuote_BNB"]
    return budget

def calculate_budget(qty_sell, qty_buy, price_sell, price_buy, t_data, retry=25, budget0=None):
    # check budget
    ok = False
    if budget0 is None:
        budget0 = init_budget(t_data)
    budget = budget0 # this should be initialised !
    if qty_sell is not None: budget += qty_sell * price_sell * t_data["priceQuote_BNB"]
    if qty_buy is not None: budget += qty_buy * price_buy * t_data["priceQuote_BNB"]
    if budget > t_data["share_BNB"]:
        ratio = t_data["share_BNB"] / budget
        if qty_sell is not None:
            qty_sell = qty_sell * ratio
            if qty_sell < t_data["lot_size"]: 
                qty_sell = t_data["lot_size"]
        if qty_buy is not None: 
            qty_buy = qty_buy * ratio
            if qty_buy < t_data["lot_size"]: 
                qty_buy = t_data["lot_size"]
        if retry > 0:
            qty_sell, qty_buy, budget, ok = calculate_budget(
                qty_sell, qty_buy, price_sell, price_buy, t_data, 
                retry=retry-1, budget0=budget0)
    else:
        ok = True
    return qty_sell, qty_buy, budget, ok


def cancel_old_orders():
    orders = lib.wallet.get_orders()
    now = lib.api.api_time()
    for order in orders:
        creation_date = pd.to_datetime(order["orderCreateTime"])
        if (now - creation_date) > datetime.timedelta(days=1):
            ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
            order_id = order["orderId"]
            pair = order["symbol"]
            lib.wallet.cancel_order(pair, order_id)
            print("{}| {} order {} canceled".format(ts, pair, order_id))  
            
def get_mask(tick):
    tick = Decimal(tick)
    mask = "{}"
    for i in range(33):
        mask = "{:0."+str(i)+"f}"
        result = mask.format(tick)
        result = Decimal(result)
        if (tick-result)==Decimal(0):
            break
    return mask

def round_by(x, scale):
    mask = get_mask(scale)
    y = int(float(x)/float(scale)) * float(scale)
    y = Decimal(mask.format(y))
    return y

def calculate_qty(qty_sell, qty_buy, price_sell, price_buy, t_data, retry=25, 
                  tokens0=None, amount0=None, budget0=None):
    # check budget
    ok = False
    if budget0 is None:
        budget0 = init_budget(t_data)
    if tokens0 is None or amount0 is None:
        balances = lib.wallet.get_balance()
        df_balance = pd.DataFrame(balances, dtype=float)
    if tokens0 is None:
        tokens0 = 0.0
        where = (df_balance["symbol"] == t_data["baseAssetName"])
        df_token = df_balance[where].copy()
        if df_token.shape[0] == 1:
            tokens0 = df_token["free"].sum()
    if amount0 is None:
        amount0 = 0.0
        where = (df_balance["symbol"] == t_data["quoteAssetName"])
        df_quote = df_balance[where].copy()
        if df_quote.shape[0] == 1:
            amount0 = df_quote["free"].sum()
        
    budget = budget0 
    budget += qty_sell * float(price_sell) * t_data["priceQuote_BNB"]
    budget += qty_buy * float(price_buy) * t_data["priceQuote_BNB"]
    tokens = qty_sell
    amount = qty_buy * float(price_buy)
    
    if budget > t_data["share_BNB"]:
        ratio = t_data["share_BNB"] / budget
        qty_sell = qty_sell * ratio
        qty_buy  = qty_buy * ratio
        if qty_sell < float(t_data["lot_size"]): 
            qty_sell = float(t_data["lot_size"])
        if qty_buy < float(t_data["lot_size"]): 
            qty_buy = float(t_data["lot_size"])
    elif tokens > tokens0:
        ratio = tokens0 / tokens
        qty_sell = qty_sell * ratio
        qty_buy  = qty_buy * ratio
    elif amount > amount0:
        ratio = amount0 / amount
        qty_sell = qty_sell * ratio
        qty_buy  = qty_buy * ratio
    else:
        ok = True
    if not ok and retry > 0:
        qty_sell, qty_buy, ok = calculate_qty(
            qty_sell, qty_buy, price_sell, price_buy, t_data, retry=retry-1, 
            tokens0=tokens0, amount0=amount0, budget0=budget0)
    
    return qty_sell, qty_buy, ok

def calculate_qty_invest(qty_sell, qty_buy, price_sell, price_buy, t_data, retry=25, 
                  tokens0=None, amount0=None):
    # check budget
    ok = False
    if tokens0 is None or amount0 is None:
        balances = lib.wallet.get_balance()
        df_balance = pd.DataFrame(balances, dtype=float)
    if tokens0 is None:
        tokens0 = 0.0
        where = (df_balance["symbol"] == t_data["baseAssetName"])
        df_token = df_balance[where].copy()
        if df_token.shape[0] == 1:
            tokens0 = df_token["free"].sum()
    if amount0 is None:
        amount0 = 0.0
        where = (df_balance["symbol"] == t_data["quoteAssetName"])
        df_quote = df_balance[where].copy()
        if df_quote.shape[0] == 1:
            amount0 = df_quote["free"].sum()
        
    tokens = qty_sell
    amount = qty_buy * float(price_buy)
    
    if tokens > tokens0:
        ratio = tokens0 / tokens
        qty_sell = qty_sell * ratio
    elif amount > amount0:
        ratio = amount0 / amount
        qty_buy  = qty_buy * ratio
    else:
        ok = True
    if not ok and retry > 0:
        qty_sell, qty_buy, ok = calculate_qty_invest(
            qty_sell, qty_buy, price_sell, price_buy, t_data, retry=retry-1, 
            tokens0=tokens0, amount0=amount0)
    
    return qty_sell, qty_buy, ok

def makeup_prices(data_price_mid, data_price_std, t_data):
    price_buy  = round_by(data_price_mid-data_price_std*RATIO, t_data["tick_size"])
    price_sell = round_by(data_price_mid+data_price_std*RATIO, t_data["tick_size"])

    return price_buy, price_sell 

def operation_loop(df):

    cancel_old_orders()

    df_tmp = df.copy()
    df_tmp = df_tmp[df_tmp["score"]>0.0]
    nb_actions = min(int(1 + df.shape[0]*0.333), df.shape[0])
    for i in range(nb_actions):
        ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        pair = lib.market.pick_symbol(df_tmp)
        d_loc = df[df["pair"]==pair].to_dict()
        t_data = {v:list(k.values())[0] for (v,k) in d_loc.items()}
        df_tmp = df_tmp.drop(df_tmp[df_tmp["pair"]==pair].index)
        df_tmp = df_tmp.reset_index(drop=True)

        print("----{}----                          ".format(pair), end="\r")

        # setup price
        data_price_mid, data_price_std = choose_price(t_data)
        price_buy, price_sell = makeup_prices(data_price_mid, data_price_std, t_data)

        # define first estimate of buy and sell quantity, including invest parameter 
        qty_sell = (0.9 + 0.1 * t_data["investQuote"])  * 0.05  * (t_data["share_BNB"] - t_data["prod_BNB"]) / t_data["priceBase_BNB"]
        qty_buy = (0.9 + 0.1 * t_data["investBase"])  * 0.05  * (t_data["share_BNB"] - t_data["prod_BNB"]) / t_data["priceBase_BNB"]
        if qty_sell < float(t_data["lot_size"]): qty_sell = float(t_data["lot_size"])
        if qty_buy < float(t_data["lot_size"]): qty_buy = float(t_data["lot_size"])
        print("quantity calculated: {}/buy and {}/sell".format(qty_buy, qty_sell ), end="\r")

        # check budget, balance, reduce if needed 
        qty_sell, qty_buy, ok = calculate_qty(
            qty_sell, qty_buy, price_sell, price_buy, t_data)
        qty_sell  = round_by(qty_sell, t_data["lot_size"])
        qty_buy = round_by(qty_buy, t_data["lot_size"])
        print("quantity meeting budget and balance: {}/buy and {}/sell".format(qty_buy, qty_sell ), end="\r")

        if ok and qty_sell>Decimal(0) and qty_buy>Decimal(0):
            mode = "Fork"
            
            print("{}| {} sell {:.08f} {} @ {:.08f} {} -> ".format(
                ts, mode, qty_sell, t_data["baseAssetName"], price_sell, t_data["quoteAssetName"]), end="")
            res = lib.wallet.send_order("sell", qty_sell, price_sell, t_data["pair"])
            print(len(res))

            print("{}| {} buy  {:.08f} {} @ {:.08f} {} -> ".format(
                ts, mode, qty_buy, t_data["baseAssetName"], price_buy, t_data["quoteAssetName"]), end="")
            res = lib.wallet.send_order("buy", qty_buy, price_buy, t_data["pair"])
            print(len(res))
        else: 
            print("{}| conditions not met for {} fork action            ".format(ts, t_data["pair"]))
            # print("INVEST_TRG", INVEST_TRG)
            # print("t_data[investQuote]", t_data["investQuote"])
            # print("t_data[investBase]",  t_data["investBase"])
            # test = ((t_data["investQuote"] > INVEST_TRG and  t_data["investQuote"] > t_data["investBase"]) or t_data["investBase"] == 0.0)
            # print("test invest quote", test)
            # test =((t_data["investBase"] > INVEST_TRG and t_data["investBase"] > t_data["investQuote"]) or t_data["investQuote"] == 0.0)
            # print("test invest base", test)

            if (t_data["investQuote"] > INVEST_TRG_HIGH and  t_data["investQuote"] > t_data["investBase"]) or (t_data["investBase"] <= INVEST_TRG_LOW and t_data["investBase"] < t_data["investQuote"]): 
                mode = "Invest (quote)"
                qty_sell = 0.05  * t_data["share_BNB"] / t_data["priceBase_BNB"]
                if qty_sell < float(t_data["lot_size"]): qty_sell = float(t_data["lot_size"])
                qty_sell, qty_buy, ok = calculate_qty_invest(
                    qty_sell, 0.0, price_sell, price_buy, t_data)
                qty_sell  = round_by(qty_sell, t_data["lot_size"])
                if ok and qty_sell>Decimal(0):
                    print("{}| {} sell {:.08f} {} @ {:.08f} {} -> ".format(
                        ts, mode, qty_sell, t_data["baseAssetName"], price_sell, t_data["quoteAssetName"]), end="")
                    res = lib.wallet.send_order("sell", qty_sell, price_sell, t_data["pair"])
                    print(len(res))

            if (t_data["investBase"] > INVEST_TRG_HIGH and t_data["investBase"] > t_data["investQuote"]) or (t_data["investQuote"] <= INVEST_TRG_LOW and t_data["investQuote"] <  t_data["investBase"] ):
                mode = "Invest (token)"
                qty_buy = 0.05  * t_data["share_BNB"] / t_data["priceBase_BNB"]
                if qty_buy < float(t_data["lot_size"]): qty_buy = float(t_data["lot_size"])
                qty_sell, qty_buy, ok = calculate_qty_invest(
                    0.0, qty_buy, price_sell, price_buy, t_data)
                qty_buy = round_by(qty_buy, t_data["lot_size"])
                if ok and qty_buy>Decimal(0):
                    print("{}| {} buy  {:.08f} {} @ {:.08f} {} -> ".format(
                        ts, mode, qty_buy, t_data["baseAssetName"], price_buy, t_data["quoteAssetName"]), end="")
                    res = lib.wallet.send_order("buy", qty_buy, price_buy, t_data["pair"])
                    print(len(res))

def main():
    df = None
    for i in range(1000000):
        try:
            if i%41 == 0 : df = None
            if df is None:
                df = lib.market.get_market_opportunities()
                ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
                print("{}| markets renewed...".format(ts))
            else:
                df = lib.market.refresh_market_opportunities(df)

            operation_loop(df)
            lib.wallet.write_file("healthcheck", "OK")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("operations error:", exc_type, exc_tb.tb_lineno, str(e))

        wait_time = 3 + random.randint(0, 7)    
        for minute in range(wait_time):
            print("{}/{}".format(minute, wait_time), end="\r")
            time.sleep(60)
    
if __name__ == "__main__":
    main()