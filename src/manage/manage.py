import os
import pandas as pd
import numpy as np
from keras.models import load_model
from src.data import buildEmbeddings, getData
from src.models import correlation, neuralNetSum
from src.preprocessor import preprocessing

def train(logger, config, options):
    if(options.toBuildEmbeddings):
        buildEmbeddings.build(logger, config)
    if(options.toPreprocess):
        getData.getCorpus(preprocess=True)
    if(options.toPreprocess or options.toBuildEmbeddings):
        correlation.correlation_dictionary()
    corpus, labels = getData.getCorpus()
    neuralNetSum.buildNN_sum(corpus, labels, logger, config)
    neuralNetSum.buildNN_sum(corpus, labels, logger, config, correlated=True)

def interactive(logger, options,file_path, correlated=False):
    if(correlated):
        model = load_model(os.path.join(file_path,'models','nn_model_correlated.h5'))
    else:
        model = load_model(os.path.join(file_path,'models','nn_model.h5'))
    while(True):
        sentence = input('Sentence (-1 to leave): ')
        if(sentence=='-1'):
            break
        sentence = [sentence]
        sentence = preprocessing.preprocess(sentence)
        sentence = neuralNetSum.transformSum(sentence, False)
        results = model.predict_classes(sentence)
        if(results[0]==1):
            print('subjetiva')
        else:
            print('objetiva')
def file_classify(logger, options, file_path, correlated=False):
    if(correlated):
        model = load_model(os.path.join(file_path,'models','nn_model_correlated.h5'))
    else:
        model = load_model(os.path.join(file_path,'models','nn_model.h5'))
    input_data = pd.read_csv(options.input, header=None, sep='\n')
    input_data = input_data.values
    input_data = [sentence[0] for sentence in input_data]
    input_data = preprocessing.preprocess(input_data)
    input_data = neuralNetSum.transformSum(input_data, correlated)
    results = model.predict_classes(input_data)
    print(results)
