import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation
from keras import backend as K

from src.data import getData
from src.models.correlation import correlation_dictionary
def transformSum(corpus, correlated, cross_validation=False):
    '''transform each sentence into the sum of embeddings'''
    word_vectors = getData.getEmbeddings().wv
    #normalize
    word_vectors.init_sims(replace=True)
    if(correlated and cross_validation):
        correlation_dictionary(corpus)
    corpus = [sentence.split(' ') for sentence in corpus]
    corpus_sum = [[0]*word_vectors.vector_size for sentence in corpus]
    if(correlated):
        file_path = os.path.split(os.path.abspath(__file__))[0]
        fp = open(os.path.join(file_path,'correlation.json'), 'r')
        correlationDic = json.load(fp)
        fp.close()
        for i, sentence in enumerate(corpus):
            corpus[i] = [correlationDic.get(word, word) for word in sentence]
        del correlationDic
    for i,sentence in enumerate(corpus):
        for word in sentence:
            try:
                corpus_sum[i] = corpus_sum[i] + word_vectors[word]
            except:
                continue
        corpus_sum[i] = np.array(corpus_sum[i])/len(sentence)
    return np.array(corpus_sum)

def buildNN_sum(corpus, labels, logger, config, cross_validation=False, correlated=False):
    file_path = os.path.split(os.path.abspath(__file__))[0]
    #transform each sentence into the sum of word vectors
    corpus = transformSum(corpus, correlated, cross_validation=cross_validation)
    #transform one dimensional into categorical for classification
    labels = np_utils.to_categorical(labels, num_classes=2)
    #neural network model -> one hidden layer with n_neurons neurons
    rnn_model = Sequential()
    rnn_model.add(Dense(config['model']['num_neurons'], input_dim=len(corpus[0]), activation='relu'))
    rnn_model.add(Dense(2, activation='softmax'))
    rnn_model.compile(loss=config['model']['loss'], optimizer=config['model']['optimizer'], metrics=['accuracy'])
    rnn_model.fit(corpus, labels, epochs=config['model']['epochs'],verbose=0)
    if(cross_validation):
        return rnn_model
    else:
        if(correlated):
            rnn_model.save(os.path.join(file_path,'nn_model_correlated.h5'))
        else:
            rnn_model.save(os.path.join(file_path,'nn_model.h5'))
