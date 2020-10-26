import pandas as pd

from decimal import Decimal

import lib
import operatorQN

def match_price(df, target, direction, tick_size):
    match_price  = float(df.head(1)["price"]) 
    offset = - Decimal(tick_size) * direction
    df["diff"] = (df["price"] - float(target)) * direction
    mask = (df["diff"] > 0)
    diff_min = df[mask]["diff"].min()
    mask = (df["diff"].abs() == diff_min)
    if df[mask].shape[0] > 0: 
        match_price  = Decimal(df[mask].head(1)["price"].mean()) 
        match_price += offset
    return operatorQN.round_by(match_price, tick_size)

def makeup_prices(data_price_mid, data_price_std, t_data):
    price_buy  = operatorQN.round_by(data_price_mid-data_price_std*operatorQN.RATIO, t_data["tick_size"])
    price_sell = operatorQN.round_by(data_price_mid+data_price_std*operatorQN.RATIO, t_data["tick_size"])

    # get order book
    symbol = t_data["pair"]
    baj = lib.api.get_rj("depth?symbol={}&limit=1000".format(symbol))

    cols = ["price", "quantity"]
    dfa = pd.DataFrame(baj["asks"], columns=cols, dtype=float)
    dfb = pd.DataFrame(baj["bids"], columns=cols, dtype=float)
    
    # match price in order book
    price_buy  = match_price (dfb, price_buy,  -1, tick_size=t_data["tick_size"]) 
    price_sell = match_price (dfa, price_sell, +1, tick_size=t_data["tick_size"]) 
    
    return price_buy, price_sell 

def main():
    operatorQN.makeup_prices = makeup_prices
    operatorQN.main()
    
if __name__ == "__main__":
    main()