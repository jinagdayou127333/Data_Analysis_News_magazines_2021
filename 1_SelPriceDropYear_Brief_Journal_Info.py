# coding=gbk
#依据单价和刊期种类筛选条目
import pandas as pd
import numpy as np
nj = pd.read_excel('Temp/Brief_Journal.xlsx')
print(nj.columns)
nj_tp=nj[nj['刊期种类']!='年刊']
nj_tp=nj_tp.dropna()
nj_price_j=nj_tp['订阅单价(元)']
ds_price=nj_price_j.describe()
print(ds_price)
three_quarters=ds_price[6]
one_quarters=ds_price[4]
IQR=ds_price[6]-ds_price[4]
upper_1=three_quarters+1.5*IQR
lower_1=one_quarters-1.5*IQR
print('BY IQR:',lower_1,upper_1)
mean_price=ds_price[1]
std_price=np.std(nj_price_j.values)
upper_2=mean_price+3*std_price
lower_2=mean_price-3*std_price
print('BY Empirical Rule:',lower_2,upper_2)
nj_clean=nj_tp[(nj_price_j>lower_2) & (nj_price_j<=upper_2)]
nj_clean1=nj_clean.iloc[:,2:]
nj_clean1.to_excel('Temp/SelPriceDropYear_Brief_Journal.xlsx')
