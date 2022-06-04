# coding=gbk
import pandas as pd
nj = pd.read_excel('Data/2021NJ.xlsx')
nj_j = nj[(nj['类别']=='期刊')]
print(nj_j.columns)
nj_brief_j=nj_j.loc[:,['报刊名称','类别','刊期种类','订阅单价(元)','年价(元)','版面/页数','报刊简介','产品分类','是否畅销报刊']]
nj_brief_j.reset_index(inplace=True)
nj_brief_j.to_excel('Temp/Brief_Journal.xlsx')

nj_n = nj[(nj['类别']=='报纸')]
nj_brief_n=nj_n.loc[:,['报刊名称','类别','刊期种类','订阅单价(元)','年价(元)','版面/页数','报刊简介','产品分类','是否畅销报刊']]
nj_brief_n.reset_index(inplace=True)
nj_brief_n.to_excel('Temp/Brief_News.xlsx')