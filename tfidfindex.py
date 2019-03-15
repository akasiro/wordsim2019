# -*- coding: utf-8 -*-


import jieba
import pandas as pd
from gensim import corpora,models,similarities
from collections import defaultdict   #用于创建一个空的字典，在后续统计词频可清理频率少的词语

stopwordsdf = pd.read_csv('stopwords.txt', index_col=False, sep='\t', quoting=3, names=['stopword'],
                          encoding='utf-8')
stopwords = stopwordsdf.stopword.values.tolist()

#分词并去掉停用词
def cut(text,stopwords = stopwords):
    generator = jieba.cut(text)
    res = [word for word in generator]
    res_clean = []
    for word in res:
        if word in stopwords or len(word) == 1:
            continue
        res_clean.append(word)
    return res_clean


def tfidfmodel(textlist):
    # 1、将【文本集】生成【分词列表】
    texts = [cut(text) for text in textlist]
    # 2、基于文本集建立【词典】，并提取词典特征数
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id.keys())
    # 3、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 4、使用【TF-IDF模型】处理语料库
    tfidf = models.TfidfModel(corpus)
    #6、对【稀疏向量集】建立【索引】
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)

    parameter = [index, tfidf,dictionary]
    return parameter

def tfidfsimilarity(text,parameter):
    # 5、同理，用【词典】把【搜索词】也转换为【稀疏向量】
    dictionary = parameter[2]
    index = parameter[0]
    tfidf = parameter[1]
    kw_vector = dictionary.doc2bow(cut(text))
    sim = index[tfidf[kw_vector]]
    return sim




if __name__ == '__main__':
    with open('testdata/446324234intro.txt','r',encoding='utf-8') as f:
        text1 = f.read()
    with open('testdata/472885640intro.txt','r',encoding='utf-8') as f:
        text2 = f.read()
    with open('testdata/680465449intro.txt','r',encoding='utf-8') as f:
        text3 = f.read()
    parameter =tfidfmodel([text1,text2])
    print(tfidfsimilarity(text3,parameter))

