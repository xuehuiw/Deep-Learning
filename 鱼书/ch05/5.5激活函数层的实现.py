import numpy as np

#ReLU层
class ReLU:
    def __init__(self):
        self.mask = None#在前向传播时保存输入的值,在反向传播时使用
        #.mask（掩码）是一个实例变量，只随对象本身变化，而不随类的变化。每个对象都有自己的.mask变量，互不干扰。
    
    def forward(self, x):
        out = x.copy()#dout是 “输出的梯度”（比如后续层传过来的梯度）。如果直接写out = x，只是给x起了个别名，改out会同步改x，这是新手常踩的坑。因此需要使用copy()方法创建一个新的数组。
        out[self.mask] = 0#将输入中小于等于0的部分置为0,大于0的部分保持不变
        return out

    def backward(self, dout):
        dout[self.mask] = 0#布尔索引，批量修改数组元素。用正向保存的self.mask，把dout里 “原 x≤0” 的位置的梯度都改成 0；大于0的部分的梯度保持不变
        dx = dout#把处理后的dout赋值给dx，dx就是 ReLU 层输入x的梯度；
        return dx

#Sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None
        #.out是一个实例变量，只随对象本身变化，而不随类的变化。每个对象都有自己的.out变量，互不干扰。

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
 #Affine层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
    