# Subjectivity Classifier for Portuguese

This project aims to classify whether a sentence express any kind of sentiment/opinion or is a objetive/non-opinative sentence using machine learning techniques.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To run this application, you will need Python >= 3.6 and to manually install this library: https://github.com/tfcbertaglia/enelvo

### Installing
The setup.py hasn't been properly configured yet, so you just have to check if you have all dependencies installed. Inside the 'subjectivity-classification' folder, run:
```
pip3 install --user -r requirements.txt
```

## Running

Inside the 'subjectivity-classification' folder, you can use with the default configuration by running:
```
python3 -m src interactive
```
In this case, you will just have to type your sentence in the command line.
You can also insert a txt file by running:
```
python3 -m src --input pathTo/inputFile.txt
```
Where --input needs the path to the inputFile. Each line in the inputFile is considered as a sentence.


### Configuration and Training

Inside the src folder there's a json file called "config.json". In this file you can change a few training configurations for the embeddings training (for details of the configuration, check the gensim Word2Vec documentation: https://radimrehurek.com/gensim/models/word2vec.html). In the same file, you can change the number of neurons, loss function, optimizer and epochs used by the neural network used for classification (for more details, check the keras documentation: https://keras.io/getting-started/sequential-model-guide/).
When you change the json file, make sure to run:
```
python3 -m train -buildEmbeddings
```
The argument "-buildEmbeddings" is necessary only if you have changed the embeddings configuration.
You can also see the results of your configuration in a 10-fold cross-validation by running:
```
python3 -m train test
```
Make sure to run -buildEmbeddings before the test if you want to see the differences in the results due to the embeddings training.

