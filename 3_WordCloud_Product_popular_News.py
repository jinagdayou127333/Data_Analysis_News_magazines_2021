# coding=gbk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imageio import imread
import re
import jieba
import collections
nj= pd.read_excel('Temp/SelPriceDropYear_Brief_News.xlsx')
print(nj.columns)
user_labels_list = []
nj_popular=nj
# nj_popular=nj[nj['是否畅销报刊']=='是']
user_labels=nj_popular.iloc[:,1:]
for label in user_labels['产品分类']:    #产品分类 #报刊简介 #报刊名称 #把表格的每一列存到列表user_labels_list中
    user_labels_list.append(label)
cleanedList = [x for x in user_labels_list if str(x) != 'nan']   #去掉空值
mytext = ''.join(cleanedList)    #把列表变成字符串
stopwords=['适合','本刊','本报','本报适合','本刊适合','简介','空白','内容简介','内容','读者对象',
           'CIP核字号','对象','读者','年','省','含','版','了','面向','突出',
           '以','的','和','是','对于','为','报纸','杂志','期刊','及','订阅','提供','等','与',
           '中国','报','我国','为主','主要','方面','类','报含',
           '中共','中央','直辖市','自治区','级','地','市','内蒙古','锡林郭勒','县级']
punctuation = """！？。＂＃＄％＆＇（）＊＋－／：；.＜＝＞＠［＼］＾＿｀｛｜｝～?????、〃》，, ( ) -《「」『』【】 '\n' 〔〕〖〗?????〝〞????–—‘'?“”??…?﹏"""
re_punctuation = "[{}]+".format(punctuation)
mytext = re.sub(re_punctuation, "", mytext)
seg_list_exact = jieba.cut(mytext, cut_all = False) # 精确模式分词
object_list = []
for word in seg_list_exact: # 循环读出每个分词
    if word not in stopwords: # 如果不在去除词库中
        if word=='宣传' or word =='报导' or word=='传播' or word=='报道':
            word='报导'
        if word=='辅导' or word=='指导' or word=='培养':
            word='培养'
        if word=='新' or word=='最新':
            word='新'
        if word=='手机' or word =='客户端':
            word='客户端'
        if word=='人教' or word=='人教版':
            word='人教版'
        if word == '科技' or word == '科学技术':
            word = '科技'
        if word == '上' or word == '半月':
            word = '上半月'
        object_list.append(word) # 分词追加到列表
word_counts = collections.Counter(object_list)
mk = imread('wordle-word-cloud.tif')   #读取词云背景形状图片
wordcloud = WordCloud(font_path = "方正粗黑宋简体.ttf",background_color="white",
                      collocations=False,max_words=2000,stopwords = stopwords,mask=mk)
wordcloud.generate_from_frequencies(word_counts)
wordcloud.to_file('Figures/依据产品分类的报纸词云.png') #保存图片

counts = word_counts.most_common(15)
print(dict(counts)) #显示词频
fig = plt.figure()
data = pd.Series(list(dict(counts).values()),index=list(dict(counts).keys()))
data.plot.bar(color='black',alpha=1,rot=80,fontsize=10)
#解决中文标签字体显示问题
plt.rcParams['font.sans-serif'] = 'SimHei' # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.title('依据产品分类的报纸前15大词汇')
plt.ylabel('频数',fontsize=12)
plt.xlabel('词汇',fontsize=12)
fig.savefig('Figures/依据产品分类的报纸前15大词云.png',dpi=300,bbox_inches='tight')