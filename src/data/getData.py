import sys
import os
import pandas as pd
import numpy as np
from src.preprocessor import preprocessing
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#informations about the xlsx file containing the corpus
SHEET = "Pesquisa"
DATA_COLUMN = 2
LABEL_COLUMN = 1

def adjust_label(labels):
    labels = [1 if x in [-1,-2, 1] else 0 for x in labels]
    return labels


def getEmbeddings(model_name="word2vecBuscape"):
    file_path = os.path.split(os.path.abspath(__file__))[0]
    try:
        model = Word2Vec.load(os.path.join(file_path,'model', model_name+".model"))
        return model
    except IOError as e:
        errno, strerror = e.args
        '''mudar para logger'''
        print("I/O error({0}): {1}".format(errno,strerror))
        print("Word2Vec model not found") 
        sys.exit()

def getCorpus(preprocess=False, filename="Computer-BR.xlsx"):
    file_path = os.path.split(os.path.abspath(__file__))[0]
    if(preprocess):
        try:
            corpus_dfs = pd.read_excel(os.path.join(file_path,'raw',filename), sheet_name=SHEET)
            corpus_dfs = corpus_dfs.values
            corpus = preprocessing.preprocess(corpus_dfs[:,DATA_COLUMN])
            labels = adjust_label(corpus_dfs[:,LABEL_COLUMN])
            np.save(os.path.join(file_path,'preprocessed',"corpus_preprocessed"), corpus)
            np.save(os.path.join(file_path,'preprocessed',"labels"), labels)
        except IOError as e:
            errno, strerror = e.args
            print("I/O error({0}): {1}".format(errno,strerror))
            sys.exit()
    else:
        try:
            corpus = np.load(os.path.join(file_path, 'preprocessed','corpus_preprocessed.npy'))
            labels = np.load(os.path.join(file_path, 'preprocessed',"labels.npy"))
        except IOError as e:
            errno, strerror = e.args
            print("I/O error({0}): {1}".format(errno,strerror))
            print("There's no preprocessed data in /data:", os.path.join(file_path,'preprocessed'))
            sys.exit()
        return corpus, labels

