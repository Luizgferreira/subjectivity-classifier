import os
import sys
import numpy as np
from gensim.models import Word2Vec

def build(logger, config, name_preprocessed = 'buscape_preprocessed.txt'):
    file_path = os.path.split(os.path.abspath(__file__))[0]
    sentences = np.loadtxt(os.path.join(file_path,'preprocessed',name_preprocessed), dtype='str', delimiter='\t')
    sentences = [sentence.split(' ') for sentence in sentences]
    model = Word2Vec(size = config['embeddings']['size_vector'],
                     min_count=config['embeddings']['min_count'],
                     workers=3, window=config['embeddings']['window'])
    model.build_vocab(sentences, progress_per=config['embeddings']['progress'])
    model.train(sentences, total_examples=len(sentences), 
                epochs=config['embeddings']['epochs'])
    model.save(os.path.join(file_path, 'model', 'word2vecBuscape.model'))
