#softmax函数的输出通过箭头与所有的输入信号相连，输出层的各个神经元都受到所有输入信号的影响
#表示：分子是输入信号ak的指数函数，分母是所有输入信号的指数函数之和
#softmax函数的输出y的值域是0~1，且所有输出的和为1，因此softmax函数的输出可以被解释为概率
#防止溢出：通过减去输入信号中的最大值来实现，输入信号的最大值为c，softmax函数的输出y仍然是正确的

import numpy as np
def softmax(a):
    c= np.max(a) #防止溢出
    exp_a= np.exp(a - c)
    sum_exp_a= np.sum(exp_a)
    y= exp_a / sum_exp_a
    return y