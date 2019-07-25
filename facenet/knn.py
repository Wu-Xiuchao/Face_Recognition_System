import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer

    dis = np.linalg.norm(np.expand_dims(x,axis=1).repeat(x_train.shape[0],axis=1)-x_train,axis=2)
    top_k_neighbor = np.argsort(dis,axis=1)[:,:k]
    y = scipy.stats.mode(y_train[top_k_neighbor],axis=1)[0]


    # end answer

    return y
