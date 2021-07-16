import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    # TODO
    mse = np.sum(np.square(Y-X@W))/(2*X.shape[0])
    # END TODO

    return mse

def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.01, max_iter=2000):
    '''
    reg - regularization parameter (lambda in Q2.1 c)
    '''
    train_mses = []
    test_mses = []

    ## TODO: Initialize W using using random normal 
    W = np.random.normal(size=(X_train.shape[1],1))
    ## END TODO

    for i in range(max_iter):

        ## TODO: Compute train and test MSE
        train_mse = mse(X_train,Y_train,W) #+ reg*(np.square(np.linalg.norm(W)))
        test_mse = mse(X_test,Y_test,W) #+ reg*(np.square(np.linalg.norm(W)))
        ## END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        ## TODO: Update w and b using a single step of gradient descent
        XT = X_train.transpose()
        W = W - lr*((XT@X_train@W-XT@Y_train)/X_train.shape[0] + (2*reg)*W)
        ## END TODO
    #W = np.linalg.inv(X_train.T @ X_train + 2 * reg * X_train.shape[0] * np.eye(X_train.shape[1])) @ X_train.T @ Y_train
    # train_mse = mse(X_train,Y_train,W) #+ reg*(np.square(np.linalg.norm(W)))
    # test_mse = mse(X_test,Y_test,W) #+ reg*(np.square(np.linalg.norm(W)))
    # train_mses.append(train_mse)
    # test_mses.append(test_mse)
    return W, train_mses, test_mses


def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.001, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    W = np.random.normal(scale=0.1,size=(X_train.shape[1],1))
    # END TODO

    for i in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.
        W_old = W.copy()
        W = W - lr*(X_train.T@X_train@W-X_train.T@Y_train)/X_train.shape[0]
        W[W>_lambda*lr] -= lr*_lambda
        W[W<-_lambda*lr] += lr*_lambda
        W[np.logical_and(W<=_lambda*lr,W>=-_lambda*lr)] = 0
        # END TODO

        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if np.linalg.norm(W-W_old) < 0.0001:
            break;
        # End TODO

    return W, train_mses, test_mses


if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    train_mses = []
    test_mses = []
    lambdas = np.arange(0,1,(1-0)/10)
    for l in lambdas:
        W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, _lambda=l)
        if l==0.1:
            W_ista = W
        #print(W)
        train_mses.append(train_mses_ista[-1])
        test_mses.append(test_mses_ista[-1])

    W_ridge, tr, te = ridge_regression(X_train,Y_train,X_test,Y_test,10)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    x_axis = np.array(np.arange(0,10))
    for i in range(10):
        lambdas[i] = "{:.1f}".format(lambdas[i])
    my_xticks = lambdas
    # plt.plot(train_mses_ista)
    # plt.plot(test_mses_ista)
    plt.figure(figsize=(16,4))
    plt.subplot(231)
    plt.xticks(x_axis, my_xticks)
    plt.plot(x_axis, train_mses)
    plt.plot(x_axis, test_mses)
    plt.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('lambdas')
    plt.ylabel('MSE')
    plt.subplot(232)
    x_axis2 = np.arange(0,X_train.shape[1])
    plt.scatter(x_axis2, W_ista, color='b', marker='.')
    plt.xlabel('weight indices')
    plt.ylabel('lasso weights')
    plt.subplot(233)
    plt.scatter(x_axis2, W_ridge, color='r', marker='.')
    plt.xlabel('weight indices')
    plt.ylabel('ridge weights')
    plt.tight_layout()
    plt.show()
    # End TODO
