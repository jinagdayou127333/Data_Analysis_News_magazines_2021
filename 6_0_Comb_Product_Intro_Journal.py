# coding=gbk
import pandas as pd
import numpy as np

nj = pd.read_excel('Temp/Final_IntroductionCluster_Journal.xls')
nj2 = pd.read_excel('Temp/Final_ProductCluster_Journal.xls')
print(nj.info())
print(nj2.info())
nj_product = nj.loc[:,['���д���','����','���','����','������','�Ƿ�������']]
nj_product['��Ʒ����']=nj2.loc[:,['��Ʒ����']]
print(nj_product.info())
nj_product.to_excel('Temp/Final_Brief_Comb_Journal.xlsx')