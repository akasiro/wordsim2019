# -*- coding: utf-8 -*-
import jieba
import numpy as np

from math import sqrt

from tfidfindex import cut

def examplewordlist(textlist):
    examplelist = [cut(text) for text in textlist]
    return examplelist

def calcos(vecter1,vector2):
    if len(vecter1) == len(vector2):
        res1 = 0
        res2 = 0
        res3 = 0
        for i in range(len(vecter1)):
            res1 = res1 + vecter1[i]*vector2[i]
            res2 = res2 + vecter1[i]*vecter1[i]
            res3 = res3 + vector2[i]*vector2[i]
        try:
            res = res1/(sqrt(res2)*sqrt(res3))
        except:
            res = np.nan
        return res
    else:
        return np.nan

def cosindex(text,examplelist):
    cosindexs = []
    text = cut(text)
    for wordlist in examplelist:
        allword = list(set(text+wordlist))
        textvector = []
        tempexamplevector = []
        for i in range(len(allword)):
            textvector.append(0)
            tempexamplevector.append(0)
        for w1 in text:
            textvector[allword.index(w1)] +=1
        for w2 in wordlist:
            tempexamplevector[allword.index(w2)] += 1
        cos = calcos(textvector,tempexamplevector)
        cosindexs.append(cos)
    return cosindexs



if __name__ == '__main__':
    with open('testdata/446324234intro.txt','r',encoding='utf-8') as f:
        text1 = f.read()
    with open('testdata/472885640intro.txt','r',encoding='utf-8') as f:
        text2 = f.read()
    with open('testdata/680465449intro.txt','r',encoding='utf-8') as f:
        text3 = f.read()

    examplelist = examplewordlist([text1,text2])
    print(cosindex(text3,examplelist))
