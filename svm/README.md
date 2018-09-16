# SVM

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

## The advantages of support vector machines are:

1) Effective in high dimensional spaces.

2) Still effective in cases where number of dimensions is greater than the number of samples.

3) Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

4) Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

## The disadvantages of support vector machines include:

1) If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

2) SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).


## Findings

1. Decision function :- RBF        C :- 1             Gamma :- "auto"               train time :- 20 mins          Accuracy :- 0.49
2. Decision function :- Linear     C :- 1             Gamma :- 1000                 train time :- 3 mins           Accuracy :- 0.98


## Problem faced

SVM with RBF is not working properly it is always giving 0.5 accuracy
