import sys
import pandas as pd

from decimal import Decimal

import lib
import lib.features

import operatorQS

def main():
    QS.SAFETY_K = 0.9
    operatorQS.main()
    
if __name__ == "__main__":
    main()