# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression

DATAPATH = '/home/ddan/Desktop/github/house_price/data/' 

train = pd.read_csv(DATAPATH+'train.csv')
test = pd.read_csv(DATAPATH+'test.csv')

#>>> train.columns
#Index([u'Id', u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
#       u'Street', u'Alley', u'LotShape', u'LandContour', u'Utilities',
#       u'LotConfig', u'LandSlope', u'Neighborhood', u'Condition1',
#       u'Condition2', u'BldgType', u'HouseStyle', u'OverallQual',
#       u'OverallCond', u'YearBuilt', u'YearRemodAdd', u'RoofStyle',
#       u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
#       u'MasVnrArea', u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual',
#       u'BsmtCond', u'BsmtExposure', u'BsmtFinType1', u'BsmtFinSF1',
#       u'BsmtFinType2', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF',
#       u'Heating', u'HeatingQC', u'CentralAir', u'Electrical', u'1stFlrSF',
#       u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
#       u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
#       u'KitchenAbvGr', u'KitchenQual', u'TotRmsAbvGrd', u'Functional',
#       u'Fireplaces', u'FireplaceQu', u'GarageType', u'GarageYrBlt',
#       u'GarageFinish', u'GarageCars', u'GarageArea', u'GarageQual',
#       u'GarageCond', u'PavedDrive', u'WoodDeckSF', u'OpenPorchSF',
#       u'EnclosedPorch', u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'PoolQC',
#       u'Fence', u'MiscFeature', u'MiscVal', u'MoSold', u'YrSold', u'SaleType',
#       u'SaleCondition', u'SalePrice'],
#      dtype='object')

# No time, I'm choosing variables randomly

feature_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','TotalBsmtSF','TotRmsAbvGrd','GarageArea','YrSold']
x_train = train[feature_cols]
x_train = x_train.fillna(0)
y_train = train.SalePrice

lm = LinearRegression()
lm.fit(x_train,y_train)

print zip(feature_cols,lm.coef_)

x_test = test[feature_cols]
x_test = x_test.fillna(0)
y_pred = lm.predict(x_test)

ss = pd.read_csv(DATAPATH+'sample_submission.csv')
del ss['SalePrice']
ss['SalePrice'] = y_pred

ss[ss < 0] = 0

ss.to_csv(DATAPATH+'sub3.csv', index=False)
