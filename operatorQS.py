import sys
import pandas as pd

from decimal import Decimal

import lib
import lib.features

import operatorQN

def makeup_prices(data_price_mid, data_price_std, t_data):
    symbol = t_data["pair"]
    
    ask, bid = 1/3, 1/3 
    try:
        ask, bid = lib.features.optimized_ask_bid(symbol)
        print("ask:{:.04f} bid:{:.04f}".format(ask, bid))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("price advisor error:", exc_type, exc_tb.tb_lineno, str(e))
    
    price_buy  = operatorQN.round_by(data_price_mid-data_price_std*bid, t_data["tick_size"])
    price_sell = operatorQN.round_by(data_price_mid+data_price_std*ask, t_data["tick_size"])

    # get order book
    symbol = t_data["pair"]
    baj = lib.api.get_rj("depth?symbol={}&limit=1000".format(symbol))

    cols = ["price", "quantity"]
    dfa = pd.DataFrame(baj["asks"], columns=cols, dtype=float)
    dfb = pd.DataFrame(baj["bids"], columns=cols, dtype=float)
    
    # match price in order book
    price_buy  = operatorQN.match_price (dfb, price_buy,  -1, tick_size=t_data["tick_size"]) 
    price_sell = operatorQN.match_price (dfa, price_sell, +1, tick_size=t_data["tick_size"]) 
    
    return price_buy, price_sell 

def main():
    operatorQN.makeup_prices = makeup_prices
    operatorQN.main()
    
if __name__ == "__main__":
    main()