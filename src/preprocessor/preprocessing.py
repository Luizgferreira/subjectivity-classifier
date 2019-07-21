import os
import re
import pyximport
import string
import numpy as np
import pandas as pd
pyximport.install()
from enelvo import normaliser

from contextlib import contextmanager

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def remove_digits(sentences):
    '''remove words containing numbers'''
    sentences = [' '.join(s for s in sentence.split()
                          if not any(c.isdigit() for c in s))
                 for sentence in sentences]
    return sentences

def remove_tab(sentences):
    '''sometimes there s \n. Using list comprehension to split and remove'''
    sentences = [' '.join(s for s in sentence.split('\n')) for sentence in sentences]
    return sentences

def remove_stopwords(sentences, stopwords):
    sentences = [' '.join(word for word in sentence.split(' ') 
                          if word not in stopwords)
                 for sentence in sentences]
    return sentences

def remove_punctuation(sentences):
    sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
    return sentences

def treat_emoticons(sentences, emoticons):
    '''a little bit confusing, but checks if each word is a emoticon
        and substitute for the word emoticon'''
    sentences = [' '.join(s if s not in emoticons
                            else 'emoticon'
                            for s in sentence.split(' '))
                for sentence in sentences]
    return sentences

def remove_whitespace(sentences):
    sentences = [re.sub(' +', ' ', sentence.strip()) for sentence in sentences]
    return sentences


def preprocess(sentences):
    file_path = os.path.split(os.path.abspath(__file__))[0]

    norm = normaliser.Normaliser()
    vnormalise = np.vectorize(norm.normalise)

    stopwords = np.genfromtxt(fname = os.path.join(file_path,'data',"stopwords.txt"), dtype='str')
    emoticons = np.genfromtxt(fname = os.path.join(file_path,'data',"emoticons.txt"), dtype='str')

    sentences = remove_tab(sentences)
    sentences = vnormalise(sentences)
    sentences = treat_emoticons(sentences, emoticons)
    sentences = remove_punctuation(sentences)
    sentences = remove_whitespace(sentences)
    return sentences
