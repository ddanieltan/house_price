# -*- coding: utf-8 -*-
import pandas as pd

DATAPATH = '/home/ddan/Desktop/github/house_price/data/' 

train = pd.read_csv(DATAPATH+'train.csv')
test = pd.read_csv(DATAPATH+'test.csv')
