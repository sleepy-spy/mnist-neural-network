# MNIST Handwritten Digit Classifier
*A fully connected neural network achieving 97.8% accuracy with L2 regularisation and gradient descent*

## Overview
This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset.

**Key Features:**
- 97.8% accuracy on Kaggle's test set
- L2 regularization
- Standard gradient descent optimisation
- Tracks the best validation accuracy during training and saves the weights for that epoch.

## Architecture
**Network Structure:**
- Input layer: 784 neurons (28x28 pixels)
- Hidden layers: 1024 -> 512
- Output layer: 10 neurons (digits 0-9)

**Hyperparameters:**
- Learning rate: 0.08
- L2 lambda: 0.001
- Dropout: 0.0
- Batch size: 64

## Dependencies
- Ensure that you have downloaded the latest version of Python
- Download the file requirements.txt to find the entire list of dependencies

## Usage
### Train a new model
python Neural_Network.py train <training_file> [yes]

### Explanation
- When training, indicate where you data is in <training_file>
- Type yes if you wish to save the training weights

### Test a model
python Neural_Network.py test <testing_file> <weights_file>

### Explanation
- When testing, you need to indicate a testing file (<testing_file>) to pull your testing data and weights file (<weights_file>) to determine which weights you wish to use

## Future Improvements
- Add batch norm to push model accuracy


