import numpy as np 
from matplotlib import pyplot as plt
import argparse

from utils import *
from p1 import mse

## ONLY CHANGE CODE BETWEEN TODO and END TODO
def prepare_data(X,degree):
    '''
    X is a numpy matrix of size (n x 1)
    return a numpy matrix of size (n x (degree+1)), which contains higher order terms
    '''
    # TODO
    X0 = X[:,0:].copy()
    X = np.ones((X.shape[0],1))
    for _ in range(degree):
        Xp = X[:,-1:].copy()
        Xp = np.multiply(X0,Xp)
        X = np.append(X,Xp,axis=1)
    # End TODO
    return X 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--degree', type=int, default=3,
                    help='Degree of polynomial to use')
    args = parser.parse_args()
    np.random.seed(42)
    degree = args.degree

    X_train, Y_train = load_data1('data3_train.csv')
    Y_train = Y_train/20
    X_test, Y_test   = load_data1('data3_test.csv')
    Y_test = Y_test/20

    X_trainp = X_train.copy()
    X_train = prepare_data(X_train,degree)
    indices_0 = np.random.choice(np.arange(200),40,replace=False)
    indices_1 = np.random.choice(np.arange(200),40,replace=False)
    indices_2 = np.random.choice(np.arange(200),40,replace=False)
    indices_3 = np.random.choice(np.arange(200),40,replace=False)

    ## TODO - compute each fold using indices above, compute weights using OLS
    X_0 = X_train[indices_0]
    Y_0 = Y_train[indices_0]
    X_1 = X_train[indices_1]
    Y_1 = Y_train[indices_1]
    X_2 = X_train[indices_2]
    Y_2 = Y_train[indices_2]
    X_3 = X_train[indices_3]
    Y_3 = Y_train[indices_3]
    W_0 = np.linalg.inv(X_0.T@X_0)@X_0.T@Y_0
    W_1 = np.linalg.inv(X_1.T@X_1)@X_1.T@Y_1
    W_2 = np.linalg.inv(X_2.T@X_2)@X_2.T@Y_2
    W_3 = np.linalg.inv(X_3.T@X_3)@X_3.T@Y_3
    ## END TODO

    X_testp = X_test.copy()
    X_test = prepare_data(X_test,degree)

    train_mse_0 = mse(X_0,Y_0,W_0)
    train_mse_1 = mse(X_1,Y_1,W_1)
    train_mse_2 = mse(X_2,Y_2,W_2)
    train_mse_3 = mse(X_3,Y_3,W_3)
    test_mse_0  = mse(X_test, Y_test, W_0)
    test_mse_1  = mse(X_test, Y_test, W_1)
    test_mse_2  = mse(X_test, Y_test, W_2)
    test_mse_3  = mse(X_test, Y_test, W_3)

    X_lin = np.linspace(X_train[:,1].min(),X_train[:,1].max()).reshape((50,1))
    X_lin = prepare_data(X_lin,degree)
    print(f'Test Error 1: %.4f Test Error 2: %.4f Test Error 3: %.4f test E 4: %.4f'%(test_mse_0,test_mse_1,test_mse_2,test_mse_3))

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(X_train[:,1],Y_train,color='orange')
    ax1.plot(X_lin[:,1],X_lin @ W_0, c='g')
    ax1.plot(X_lin[:,1],X_lin @ W_1, c='r')
    ax1.plot(X_lin[:,1],X_lin @ W_2, c='b')
    ax1.plot(X_lin[:,1],X_lin @ W_3, color='purple')
    ax1.plot(X_lin[:,1],X_lin @(W_1+W_2+W_3+W_0)/4, color='black')


    testmse0 = []
    trainmse0 = []
    testmse1 = []
    trainmse1 = []
    testmse2 = []
    trainmse2 = []
    testmse3 = []
    trainmse3 = []

    for i in range(1,7):

        X_train = prepare_data(X_trainp,i)
        indices_0 = np.random.choice(np.arange(200),40,replace=False)
        indices_1 = np.random.choice(np.arange(200),40,replace=False)
        indices_2 = np.random.choice(np.arange(200),40,replace=False)
        indices_3 = np.random.choice(np.arange(200),40,replace=False)

        X_0 = X_train[indices_0]
        Y_0 = Y_train[indices_0]
        X_1 = X_train[indices_1]
        Y_1 = Y_train[indices_1]
        X_2 = X_train[indices_2]
        Y_2 = Y_train[indices_2]
        X_3 = X_train[indices_3]
        Y_3 = Y_train[indices_3]
        W_0 = np.linalg.inv(X_0.T@X_0)@X_0.T@Y_0
        W_1 = np.linalg.inv(X_1.T@X_1)@X_1.T@Y_1
        W_2 = np.linalg.inv(X_2.T@X_2)@X_2.T@Y_2
        W_3 = np.linalg.inv(X_3.T@X_3)@X_3.T@Y_3

        X_test = prepare_data(X_testp,i)

        train_mse_0 = mse(X_0,Y_0,W_0)
        train_mse_1 = mse(X_1,Y_1,W_1)
        train_mse_2 = mse(X_2,Y_2,W_2)
        train_mse_3 = mse(X_3,Y_3,W_3)
        test_mse_0  = mse(X_test, Y_test, W_0)
        test_mse_1  = mse(X_test, Y_test, W_1)
        test_mse_2  = mse(X_test, Y_test, W_2)
        test_mse_3  = mse(X_test, Y_test, W_3)

        testmse0.append(test_mse_0)
        trainmse0.append(train_mse_0)
        testmse1.append(test_mse_1)
        trainmse1.append(train_mse_1)
        testmse2.append(test_mse_2)
        trainmse2.append(train_mse_2)
        testmse3.append(test_mse_3)
        trainmse3.append(train_mse_3)


    
    f2 = plt.figure(figsize=(16,4))
    ax2 = f2.add_subplot(241)
    ax3 = f2.add_subplot(242)
    ax4 = f2.add_subplot(243)
    ax5 = f2.add_subplot(244)
    x_axis = np.arange(1,7)
    ax2.plot(x_axis,trainmse0)
    ax2.plot(x_axis,testmse0)
    ax2.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('degree')
    plt.ylabel('MSE')
    ax3.plot(x_axis,trainmse1)
    ax3.plot(x_axis,testmse1)
    ax3.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('degree')
    plt.ylabel('MSE')
    ax4.plot(x_axis,trainmse2)
    ax4.plot(x_axis,testmse2)
    ax4.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('degree')
    plt.ylabel('MSE')
    ax5.plot(x_axis,trainmse3)
    ax5.plot(x_axis,testmse3)
    ax5.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('degree')
    plt.ylabel('MSE')
    plt.show()

