#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2  08:22:20 2020

@author: Lincoln
"""


# step 1
import nltk
import string
import os
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def get_tokens(f):
    with open(f, 'r', encoding='utf-8', errors='ignore') as article:
        text = article.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(string.punctuation)
        tokens = nltk.word_tokenize(no_punctuation)
        return tokens


def get_tf(w):
    tf = []
    file = ['art1.dat', 'art2.dat', 'art3.dat', 'art4.dat', 'art5.dat', 'art6.dat']
    for f in file:
        tokens = get_tokens(f)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        stemmer = PorterStemmer()
        stemmed = stem_tokens(filtered, stemmer)
        count = Counter(stemmed)
        tf.append(count[w]/len(stemmed))
    return tf

def get_idf(tf):
    idf = 0
    for i in tf:
        if i>0:
            idf = idf + 1
    return math.log(6/idf)
    
    
def get_tf_idf(tf, idf):
    tfidf = []
    for i in tf:
        tfidf.append(i*idf)
    return tfidf
    
    
    
    

wrd = ["win", "ringgit", "trade", "game", "kill"]


l = get_tf(wrd[0])
idf = get_idf(l)
tfidf = get_tf_idf(l,idf)
print(l)
print(idf)
print(tfidf)




"""
    print(count['win'])
    print(count['ringgit'])
    print(count['trade'])
    print(count['game'])
    print(count['killed'])
    
    print()

"""
