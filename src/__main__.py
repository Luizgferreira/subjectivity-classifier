import logging
import json
import argparse
import os
from src.data import  getData
from src.models import correlation, neuralNetSum
from src.preprocessor import preprocessing
from src.evaluation import crossValidation
from src.manage import manage
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_options():
    parser = argparse.ArgumentParser(prog='subjectivity-classification')
    parser.add_argument('choice', choices=['train', 'test', 'classify', 'interactive'])
    parser.add_argument('--input', required=False, help='input .txt file')
    parser.add_argument('--output', required=False, nargs='?',const='output.txt', default='output.txt')
    parser.add_argument('-buildEmbeddings', dest='toBuildEmbeddings', action='store_true')
    parser.add_argument('-preprocess', dest='toPreprocess', action='store_true')
    parser.add_argument('-correlated', dest='toCorrelate', action='store_true')
    args = parser.parse_args()
    if(args.choice == 'classify' and args.input is None):
        parser.error("input file is required")
    return args

def load_config(file_path):
    '''improve this for better config'''
    with open(os.path.join(file_path,"config.json"))  as json_file:
        data = json.load(json_file)
    return data

def run():
    '''get options, config, logger and run option'''
    file_path = os.path.split(os.path.abspath(__file__))[0]
    options = load_options()
    config = load_config(file_path)
    logging.basicConfig(filename='loggingInfo.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Starting logging with options: \n{} \n config:\n{}".format(options, config))
    if(options.choice=='train'):
        manage.train(logger, config, options)
    elif(options.choice=='test'):
        '''        
        acc = 0
        for i in range(50,450, 10):
            config['model']['num_neurons']=i
            for j in range(6, 50, 2):
                config['model']['epochs']=j
                res = crossValidation.CV(logger, config, correlated=options.toCorrelate)
                if(res['accuracy'][0]>acc):
                    acc = res['accuracy'][0]
                    j_final = j
                    i_final = i
                    print('acc: '+str(acc)+' neuron: '+str(i)+' epochs: '+str(j))
        print(res_final)
        print('neuron: ', i)
        print('epochs:', j)
        '''
        print(crossValidation.CV(logger, config, correlated=options.toCorrelate))
    elif(options.choice=='classify'):
        manage.file_classify(logger, options, file_path, correlated=options.toCorrelate)
    elif(options.choice=='interactive'):
        manage.interactive(logger, options, file_path, correlated=options.toCorrelate)



if __name__ == '__main__':
    run()
