# coding=gbk
import pandas as pd
import numpy as np

nj = pd.read_excel('Temp/Final_IntroductionCluster_Journal.xls')
nj2 = pd.read_excel('Temp/Final_ProductCluster_Journal.xls')
print(nj.info())
print(nj2.info())
nj_product = nj.loc[:,['发行次数','单价','年价','版面','简介聚类','是否畅销报刊']]
nj_product['产品聚类']=nj2.loc[:,['产品聚类']]
print(nj_product.info())
nj_product.to_excel('Temp/Final_Brief_Comb_Journal.xlsx')