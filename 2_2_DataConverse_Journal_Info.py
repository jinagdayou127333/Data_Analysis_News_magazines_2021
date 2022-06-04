# coding=gbk
#填充"报刊简介"中的缺失值，将"刊期种类"等转化为数字
import pandas as pd
import numpy as np
nj = pd.read_excel('Temp/SelPriceDropYear_Brief_Journal.xlsx')
print(nj.columns)
kys=['半年刊','季刊','双月刊','月刊','旬刊','半月刊','双周刊','周刊']
vls=[2,4,6,12,36,24,26,52]
kq=np.zeros([nj.shape[0],1])
for id in range(nj.shape[0]):
    for id2 in range(8):
        if nj.iloc[id,:]['刊期种类']==kys[id2]:
            kq[id]=vls[id2]
nj['发行次数']=kq
for id in range(nj.shape[0]):
    if nj.iloc[id,:]['报刊简介']=='':
        nj.iloc[id,:]['报刊简介']= nj.iloc[id,:]['报刊名称']
nj_clean2=nj.loc[:,['报刊名称', '类别', '发行次数','订阅单价(元)', '年价(元)', '版面/页数', '报刊简介',
       '产品分类', '是否畅销报刊']]
nj_clean2.rename(columns={'订阅单价(元)':'单价','年价(元)':'年价','版面/页数':'版面'},inplace=True)
nj_clean2.to_excel('Temp/Final_Brief_Journal.xlsx')
