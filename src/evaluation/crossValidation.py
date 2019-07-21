import sys
import os
import random
import logging
import json
import numpy as np

from statistics import mean, stdev
from sklearn import metrics
from src.data.getData import getCorpus, getEmbeddings
from src.models import correlation
from src.models.neuralNetSum import buildNN_sum, transformSum

def calculate_results(results, y_test, resultsDict):
    acc = metrics.accuracy_score(y_test, results)
    resultsDict.setdefault('accuracy',[]).append(acc)
    label = 'objective'
    for label_target in range(2):
        f1 = metrics.f1_score(y_test, results, pos_label=label_target)
        rec = metrics.recall_score(y_test, results, pos_label=label_target)
        prec = metrics.precision_score(y_test, results, pos_label=label_target)
        '''add to the dictionary'''
        resultsDict.setdefault('f1_'+label,[]).append(f1)
        resultsDict.setdefault('recall_'+label,[]).append(rec)
        resultsDict.setdefault('precision_'+label,[]).append(prec)
        label = 'subjective'
    return resultsDict

def randomize_split(k):
    '''randomize and generate k parts'''
    np.random.seed(1)
    X, Y = getCorpus(preprocess=False)
    data = list(zip(X,Y))
    np.random.shuffle(data)
    X,Y = zip(*data)
    X = np.array_split(X,k)
    Y = np.array_split(Y,k)
    X = [list(fold) for fold in X]
    Y = [list(fold) for fold in Y]
    return X, Y


def CV(logger, config, k=10, correlated=None):
    logger.info("Starting cross validation function")
    X_folds, Y_folds = randomize_split(k)
    #randomize and generate k folds
    resultsDict = {}
    for i in range(k):
        X_train = list()
        y_train = list()
        for j in range(k):
            if(j != i):
                X_train = X_train + X_folds[j].copy()
                y_train = y_train + Y_folds[j].copy()
        X_test = X_folds[i].copy()
        y_test = Y_folds[i].copy()
        model = buildNN_sum(X_train, y_train, logger, config, cross_validation=True, correlated=correlated)
        X_test = transformSum(X_test, correlated)
        results = model.predict_classes(X_test, verbose=0)
        resultsDict = calculate_results(results, y_test, resultsDict)
    '''save results'''
    file_path = os.path.split(os.path.abspath(__file__))[0]
    fp = open(os.path.join(file_path,'results.json'), 'w')
    json.dump(resultsDict, fp)
    fp.close()
    for key, value in resultsDict.items():
        resultsDict[key] = (mean(value), stdev(value))
    print('(mean, stdev)')
    print(resultsDict)
    logger.info("Finished cross validation")
