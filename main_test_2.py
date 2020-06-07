"""
Created on Tue Apr  2 08:22:20 2020

@author: Lincoln
"""



import nltk
import string
import os
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

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
    
    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def k_means(tfs):
    true_k = 2
    model = KMeans(n_clusters = true_k, init='k-means++', max_iter=50, n_init=1)
    model.fit(tfs)
    print('Top terms per cluster:')
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    
    for i in range(true_k):
        print("\nCluster %d: " % i)
        for ind in order_centroids[i, :10]:
            print('%s' % terms[ind])
    

wrd = ["win", "ringgit", "trade", "game", "kill"]

print('tf, idf and tf-idf for the following 5 different terms - win, ringgit, trade, game and kill\n')

for w in wrd:
    tf = get_tf(w)
    idf = get_idf(tf)
    tfidf = get_tf_idf(tf,idf)
    c = 1   
    for i in range(len(tf)):
        print('art', c, ': tf(', w, ') = ', tf[i])
        c = c+1
    
    print('\nidf(', w, ') = ', idf, '\n')
    
    c = 1   
    for i in range(len(tf)):
        print('art', c, ': tf-idf(', w, ') = ', tfidf[i])
        c = c+1
    
    print('\n\n')

print('K-Means Clustering--\n')

stemmer = PorterStemmer()

path = '/Users/user/Documents/Documents/Study/Courses/7th sem/CSC 4309 NATURAL LANGUAGE PROCESSING/NLP Machine learning'

token_dict = {}

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        article = open(file_path, 'r', encoding='utf-8', errors='ignore')
        text = article.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(string.punctuation)
        tokens = nltk.word_tokenize(no_punctuation)
        token_dict[file] = no_punctuation


tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
k_means(tfs)