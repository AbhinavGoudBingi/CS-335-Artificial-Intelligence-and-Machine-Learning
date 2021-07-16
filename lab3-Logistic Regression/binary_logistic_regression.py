import numpy as np
import argparse
from utils import *


class BinaryLogisticRegression:
    def __init__(self, D):
        """
        D - number of features
        """
        self.D = D
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        X - numpy array of shape (N, D)
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        T = X@(self.weights)
        T = 1/(1+np.exp(-T))
        T[T>=0.5] = 1
        T[T<0.5] = 0
        return T
        # END TODO

    def train(self, X, Y, lr=0.5, max_iter=10000):
        for i in range(max_iter):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            #print(X.shape[0],X.shape[1],self.weights.shape[0])
            gradient = (X.T@(Y-1/(1+np.exp(-X@self.weights))))/X.shape[0]#np.sum((Y-1/(1+np.exp(-X@self.weights)))*X,axis=0).reshape(-1,1)/X.shape[0]
            self.weights = self.weights + lr*(gradient)
            # END TODO

            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            if np.linalg.norm(gradient)<0.0001:
                return
            # End TODO

    def accuracy(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        accuracy = ((preds == Y).sum()) / len(preds)
        return accuracy

    def f1_score(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        # TODO: calculate F1 score for predictions preds and true labels Y
        indices = Y == 1
        recall = np.count_nonzero(preds[indices] == 1)/np.count_nonzero(indices)
        indices = preds == 1
        precision = np.count_nonzero(Y[indices] == 1)/np.count_nonzero(indices)
        f1 = 2*recall*precision/(recall+precision)
        return f1
        # End TODO


if __name__ == '__main__':
    np.random.seed(335)

    X, Y = load_data('data/songs.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    D = X_train.shape[1]

    lr = BinaryLogisticRegression(D)
    lr.train(X_train, Y_train)
    preds = lr.predict(X_test)
    acc = lr.accuracy(preds, Y_test)
    f1 = lr.f1_score(preds, Y_test)
    print(f'Test Accuracy: {acc}')
    print(f'Test F1 Score: {f1}')
