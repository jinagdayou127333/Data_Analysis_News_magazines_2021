# coding=gbk
import pandas as pd
nj = pd.read_excel('Data/2021NJ.xlsx')
nj_j = nj[(nj['���']=='�ڿ�')]
print(nj_j.columns)
nj_brief_j=nj_j.loc[:,['��������','���','��������','���ĵ���(Ԫ)','���(Ԫ)','����/ҳ��','�������','��Ʒ����','�Ƿ�������']]
nj_brief_j.reset_index(inplace=True)
nj_brief_j.to_excel('Temp/Brief_Journal.xlsx')

nj_n = nj[(nj['���']=='��ֽ')]
nj_brief_n=nj_n.loc[:,['��������','���','��������','���ĵ���(Ԫ)','���(Ԫ)','����/ҳ��','�������','��Ʒ����','�Ƿ�������']]
nj_brief_n.reset_index(inplace=True)
nj_brief_n.to_excel('Temp/Brief_News.xlsx')