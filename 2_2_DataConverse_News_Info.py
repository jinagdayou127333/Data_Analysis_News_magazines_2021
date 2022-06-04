# coding=gbk
#���"�������"�е�ȱʧֵ����"��������"��ת��Ϊ����
import pandas as pd
import numpy as np
nj = pd.read_excel('Temp/SelPriceDropYear_Brief_News.xlsx')
print(nj.columns)
kys=['�ձ�','��6��','��3��','��5��','�ܱ�','��4��','��2��','���±�','Ѯ2��']
vls=[365,312,156,260,52,208,104,24,72]
kq=np.zeros([nj.shape[0],1])
for id in range(nj.shape[0]):
    for id2 in range(8):
        if nj.iloc[id,:]['��������']==kys[id2]:
            kq[id]=vls[id2]
nj['���д���']=kq
for id in range(nj.shape[0]):
    if nj.iloc[id,:]['�������']=='':
        nj.iloc[id,:]['�������']= nj.iloc[id,:]['��������']
nj_clean2=nj.loc[:,['��������', '���', '���д���','���ĵ���(Ԫ)', '���(Ԫ)', '����/ҳ��', '�������',
       '��Ʒ����', '�Ƿ�������']]
nj_clean2.rename(columns={'���ĵ���(Ԫ)':'����','���(Ԫ)':'���','����/ҳ��':'����'},inplace=True)
nj_clean2.to_excel('Temp/Final_Brief_News.xlsx')
