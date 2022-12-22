# Implementing Decision Trees from Scratch

This repository contains the code for a decision tree algorithm used to determine the location of a user based on WiFi signal strengths collected from a mobile phone. The project was completed as part of the Introduction to Machine Learning module at Imperial College London.

## Getting Started ##

To run the code in this repository, you will need to have the following dependencies installed:
  - Numpy
  - Matplotlib

You can install these dependencies by running the following command:
```python
pip install numpy matplotlib
```
You can then clone this repository and navigate to the local directory:
```python
git clone https://github.com/alexmkv01/Decision_Trees.git
cd Decision_Trees
```
## Data ##
The datasets used to test the Decision Tree implementation in this project can be found in the "WIFI db" directory and are called "clean dataset.txt" and "noisy dataset.txt". Each sample is composed of 7 WiFi signal strength readings and the last column indicates the room number in which the user is standing (i.e., the label of the sample).

## Decision Tree Learning ##
Decision Tree Learning
To create the decision tree, we have implemented a recursive method **`decision_tree_learning()`** inside the **`DecisionTree`** class, which takes as arguments a matrix containing the dataset and a depth variable (used to compute the maximal depth of the tree).  The function **`FIND_SPLIT()`** is used to choose the attribute and value that result in the highest information gain.

## Evaluation ##
We have implemented an evaluation method **`evaluate()`**  inside the **`DecisionTree`** class, which takes a trained tree and a test dataset and returns the accuracy of the tree. We have also performed 10-fold cross validation on both the clean and noisy datasets and reported the following classification metrics for both datasets:

  - Confusion matrix
  - Accuracy
  - Recall and precision rates per class
  - F1-measures derived from the recall and precision rates

## Pruning and Evaluation ## 
To reduce the performance difference between the clean and noisy datasets, we have implemented a pruning function **`prune()`**  inside the **`DecisionTree`** class, based on reducing the validation error. The pruned decision tree is then re-evaluated using the same metrics as before.

## Running the Code ## 
To run the code, simply open the Jupyter notebook file and run the cells in order. Make sure that the datasets are in the same local directory as the notebook file.

## Results ##
The results of our experiments can be found in the report included in this repository. In our implementation, we were able to achieve near full marks for this coursework.
