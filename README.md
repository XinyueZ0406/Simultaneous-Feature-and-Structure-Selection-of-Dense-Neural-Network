# Simultaneous-Feature-and-Structure-Selection-of-Dense-Neural-Network

## Introduction
This is an implementation of the paper "Simultaneous Feature and Structure Selection of Dense Neural Network".
An effective neural network selection method that can simultaneously select features and structure is proposed in this paper. This
method starts with a dense neural network and then uses two-step
connection selection to efficiently obtain a sparse neural network
architecture with the useful features and the optimal structure. The
first step uses LASSO to select the input links to each node. The
second step cycles over all links one at a time, removing those that
do not sufficiently improve the model fit. 
In this code, it includes three important steps: DNN training, backward lasso selection, and hard threshold method.
And it can be used in regression, binary, and three-label classification problems.


## Usage
1. Install Tensorflow, glmnet 
2. As an example, the following command implementate feature and structure selection on IRIS data set starting with 10 hidden layers.
```
run main.py
```
3. Note that LASSO selection step took about 15 minutes to converge, while the hard thresholding step could take as much as two hours.
4. Options:
    1. h (the number of hidden layers of DNN): you can choose different h based on your dataset size or the complexity of your problem. Note that the larger h, the more running time and memory.
    2. $\lambda$ in LASSO selection: here we give two options: $\lambda_{min}$ that minimizes cross-validated error and $\lambda_{1se}$
    that is the largest $\lambda$ value for which the cross-validated error is within one standard deviation of the minimum. Usually, $\lambda_{1se}$ can give sparser model.
    3. cut-off value $\theta$: it can be chosen by cross validation. Due to the computational cost of our method, we prefer to use a single validation set.   Usually, the larger $\theta$, the sparser the model.
