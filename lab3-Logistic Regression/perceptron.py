import numpy as np
import argparse

def get_data(dataset):
    datasets = ['D1', 'D2']
    assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
    X_train = np.loadtxt(f'data/{dataset}/training_data')
    Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
    X_test = np.loadtxt(f'data/{dataset}/test_data')
    Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

    return X_train, Y_train, X_test, Y_test

def get_features(x):
    '''
    Input:
    x - numpy array of shape (2500, )

    Output:
    features - numpy array of shape (D, ) with D <= 5
    '''
    ### TODO
    x = x.reshape(50,50)
    sub_shape = (3,3)
    view_shape = tuple(np.subtract(x.shape, sub_shape) + 1) + sub_shape
    strides = x.strides + x.strides
    sub_matrices = np.lib.stride_tricks.as_strided(x,view_shape,strides)
    conv_filter = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    m = np.einsum('ij,lkij->kl',conv_filter,sub_matrices)
    perimeter = np.count_nonzero(m)
    area = np.count_nonzero(x)
    maxi = 0
    mini = 50
    for i in range(50):
        if x[25][i] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[i][25] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[25][49-i] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[49-i][25] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[i][i] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[49-i][49-i] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[i][49-i] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    for i in range(50):
        if x[49-i][i] == 1:
            if maxi < i:
                maxi = i
            elif mini > i:
                mini = i
            break
    #print(np.array([1,area/5000,(perimeter**2)/(area*100),(maxi-mini)/30]))
    return np.array([1,area/5000,(perimeter**2)/(area*100),(maxi-mini)/30])
    ### END TODO

class Perceptron():
    def __init__(self, C, D):
        '''
        C - number of classes
        D - number of features
        '''
        self.C = C
        self.weights = np.zeros((C, D))
        
    def pred(self, x):
        '''
        x - numpy array of shape (D,)
        '''
        ### TODO: Return predicted class for x
        x.reshape(-1,1)
        return np.argmax(self.weights@x)
        ### END TODO

    def train(self, X, Y, max_iter=10):
        for iter in range(max_iter):
            for i in range(X.shape[0]):
                ### TODO: Update weights
                fp = self.pred(X[i])
                if fp != int(Y[i]):
                    self.weights[fp] -= X[i]
                    self.weights[int(Y[i])] += X[i]
                ### END TODO
            #print(f'Train Accuracy at iter {iter} = {self.eval(X, Y)}')

    def eval(self, X, Y):
        n_samples = X.shape[0]
        correct = 0
        for i in range(X.shape[0]):
            if self.pred(X[i]) == Y[i]:
                correct += 1
        return correct/n_samples

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_data('D2')

    X_train = np.array([get_features(x) for x in X_train])
    X_test = np.array([get_features(x) for x in X_test])

    C = max(np.max(Y_train), np.max(Y_test))+1
    D = X_train.shape[1]

    perceptron = Perceptron(C, D)

    perceptron.train(X_train, Y_train)
    acc = perceptron.eval(X_test, Y_test)
    print(f'Test Accuracy: {acc}')
