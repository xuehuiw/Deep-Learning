import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c= np.max(a) #防止溢出
    exp_a= np.exp(a - c)
    sum_exp_a= np.sum(exp_a)
    y= exp_a / sum_exp_a
    return y

def ReLU(x):
    return np.maximum(0, x)