import os
import sys
import json
import numpy as np
from collections import Counter
from src.data.getData import getEmbeddings, getCorpus
from gensim.models import Word2Vec



def find(dicTest):
    '''create a list with repetitions and use counter to get the most
        frequent word'''
    allWords = [word for key_word, cor_words in dicTest.items()
                for word in cor_words]
    countWords = Counter(allWords)
    return countWords.most_common(1)[0][0]

def get_topn(model, corpus, n=5):
    '''get the n most similar words for each word'''
    dic_top = {}
    all_words = list(set(word for sentence in corpus for word in sentence.split(' ')))
    for word in all_words:
        if(word in model.wv.vocab):
            #next step: limit vocabulary to corpus
            x = model.wv.similar_by_word(word, topn=n)
            x = [k[0] for k in x]
            dic_top[word] = x
        else:
            #if the model vocabulary doesn't contain "word" it maps itself
            dic_top[word] = [word]
    return dic_top


def cut(word, dic_cut, result_dic):
    for vocab in dic_cut:
        if(word in dic_cut[vocab]):
            result_dic[vocab] = word
            dic_cut[vocab] = None
    return dic_cut, result_dic

def manage(dic_topn):
    dic_cut = dic_topn.copy()
    result_dic = {}
    while(True):
        if(len(dic_cut) > 0):
            word = find(dic_cut)
            dic_cut, result_dic = cut(word, dic_cut, result_dic)
        else:
            break
        dic_cut = {word: pairs for word, pairs in dic_cut.items()
                  if pairs != None}
    return result_dic

def correlation_dictionary(sentences=None, cross_validation=False):
    file_path = os.path.split(os.path.abspath(__file__))[0]
    model = getEmbeddings()
    #get a dictionary that maps to every word in the corpus its five most similar 
    #words based in the word2vec model
    if(sentences is None):
        sentences, _ = getCorpus()
    dic_topn = get_topn(model, sentences, n=5)
    result_dic = manage(dic_topn)
    if(cross_validation):
        return result_dic
    else:
        fp = open(os.path.join(file_path,'correlation.json'), 'w')
        json.dump(result_dic, fp)
        fp.close()
