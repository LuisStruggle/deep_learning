import numpy as np

# from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    用循环实现的低效率版本的互熵损失函数和梯度求解
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[1]
    num_classes = W.shape[0]

    fs = W.dot(X)
    for i in range(num_train):
        f = fs[:, i]
        # shift the f value to achieve numerial stability
        f -= np.max(f)

        # probability interpretation for each class
        p = np.exp(f) / np.sum(np.exp(f), axis=0)

        # p = np.exp(f[y[i]]) / np.sum( f, axis=0 )
        loss += -f[y[i]] + np.log(np.sum(np.exp(f), axis=0))

        for j in range(num_classes):
            dW[j, :] += p[j] * X[:, i]

        dW[y[i], :] -= X[:, i]
        ##################################################
        # So the gradient in total is:
        # dW = (p[j] - 1) * X[:, i]     , if j == i
        #    =  p[j]      * X[:, j]     , otherwise
        #
        # just use chain rule to calculate gradient
        ##################################################

    # calculate average batch gradient
    dW /= num_train
    # calculate average batch loss
    loss /= num_train

    # regularization
    loss += 0.5 * reg * np.sum(W**2)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  向量化之后的损失函数和梯度求解
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[1]
    # num_classes = W.shape[0]

    f = W.dot(X)
    f -= np.amax(f, axis=0)

    prob = np.exp(f) / np.sum(np.exp(f), axis=0)
    prob[y, range(num_train)] -= 1

    # Data loss
    loss = np.sum(-f[y, range(num_train)] + np.log(np.sum(np.exp(f), axis=0))
                  ) / num_train

    # Weight sub-gradient
    dW = np.dot(X.T) / num_train

    # Regularization term
    loss += 0.5 * reg * np.sum(W**2)
    dW += reg * W
    return loss, dW
