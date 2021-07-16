import numpy as np 

def linear_kernel(X,Y,sigma=None):
    '''Returns the gram matrix for a linear kernel
    
    Arguments:
        X - numpy array of size n x d
        Y - numpy array of size m x d
        sigma - dummy argment, don't use
    Return:
        K - numpy array of size n x m
    ''' 
    # TODO 
    # NOTE THAT YOU CANNOT USE FOR LOOPS HERE
    return X@Y.T
    pass
    # END TODO

def gaussian_kernel(X,Y,sigma=0.1):
    '''Returns the gram matrix for a rbf
    
    Arguments:
        X - numpy array of size n x d
        Y - numpy array of size m x d
        sigma - The sigma value for kernel
    Return:
        K - numpy array of size n x m
    '''
    # TODO
    # NOTE THAT YOU CANNOT USE FOR LOOPS HERE
    return np.exp((-(np.linalg.norm(Y[None,:,:]-X[:,None,:],axis=2)**2))/(2*(sigma**2)))
    pass
    # END TODO

def my_kernel(X,Y,sigma):
    '''Returns the gram matrix for your designed kernel
    
    Arguments:
        X - numpy array of size n x d
        Y - numpy array of size m x d
        sigma- dummy argment, don't use
    Return:
        K - numpy array of size n x m
    ''' 
    # TODO
    # NOTE THAT YOU CANNOT USE FOR LOOPS HERE     
    return np.power(X@Y.T+1,4)
    pass
    # END TODO
