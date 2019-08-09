# Subjectivity Classifier for Portuguese

This project aims to classify if a sentence express some kind of sentiment using machine learning techniques.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To run this application, you will need Python >= 3.6 and manually install this library: https://github.com/tfcbertaglia/enelvo

### Installing
The setup.py hasn't been set up correctly yet, so you just have to check that you have the dependencies installed. To do this, inside the folder 'subjectivity-classification' run:

```
pip3 install --user -r requirements.txt
```

## Running

Inside the 'subjectivity-classification' folder, you can use it with the default configuration by running:
```
python3 -m src interactive
```
In this case, you only need to type your sentence on the command line.
You can also insert a txt file by running:
```
python3 -m src --input pathTo/inputFile.txt
```
Where --input needs the path to the inputFile. Each line of the inputFile is considered a sentence.


### Configuration and Training

Inside the 'src' folder there is a 'json' file named 'config.json'. In this file it is possible to change some training settings of the neural network and the words embeddings. For more details on the meaning of each embeddings training field, check the gensim documentation: https://radimrehurek.com/gensim/models/word2vec.html. For the neural network, check the keras documentation: https://keras.io/getting-started/sequential-model-guide/
When you change the json file, be sure to run:
```
python3 -m train -buildEmbeddings
```
The argument "-buildEmbeddings" is necessary only if you have changed the embeddings configuration.
You can also see the results of your configuration in a 10-fold cross-validation by running:
```
python3 -m train test
```
Make sure to run -buildEmbeddings before the test if you want to see the differences in the results due to the embeddings training.

