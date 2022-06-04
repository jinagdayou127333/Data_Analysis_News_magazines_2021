# coding=gbk
#依据单价筛选条目
import pandas as pd
import numpy as np
nj = pd.read_excel('Temp/Brief_Journal.xlsx')
print(nj.columns)
nj_price_j=nj['订阅单价(元)']
ds_price=nj_price_j.describe()
print(ds_price)
three_quarters=ds_price[6]
one_quarters=ds_price[4]
IQR=ds_price[6]-ds_price[4]
upper_1=np.floor(three_quarters+1.5*IQR)
lower_1=np.ceil(one_quarters-1.5*IQR)
print('BY IQR:',lower_1,upper_1)
mean_price=ds_price[1]
std_price=np.std(nj_price_j.values)
upper_2=np.floor(mean_price+3*std_price)
lower_2=np.ceil(mean_price-3*std_price)
print('BY Empirical Rule:',lower_2,upper_2)
nj_clean=nj[(nj_price_j>lower_2) & (nj_price_j<=upper_2)]
nj_clean1=nj_clean.iloc[:,2:]
nj_clean1.to_excel('Temp/SelPrice_Brief_Journal.xlsx')
