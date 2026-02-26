#03 Back Propagation
##反向传播是一种计算神经网络中权重梯度的方法。
###它通过链式法则将损失函数对输出的梯度逐层传递回输入层，从而计算出每个权重的梯度。这些梯度可以用来更新权重，以最小化损失函数。
###他相当于一个计算器，输入一个损失函数和一个模型参数，它会自动计算出这个参数的梯度。这样我们就不需要手动计算梯度了，可以更快地训练模型。
import torch
x_data=[1.0, 2.0, 3.0]
y_data=[2.0, 4.0, 6.0]
w=torch.tensor([1.0], requires_grad=True)

def forward(x):
    return w*x

def loss(x, y):
    y_pred=forward(x)
    return (y_pred-y)**2

print('Predict(before training)', 4, forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data, y_data):
        l=loss(x,y)
        l.backward()  # 反向传播，计算w的梯度
        print('\tgrad:', x, y, w.grad.item()) 
        w.data -= 0.01 * w.grad.data  # 更新权重
    w.grad.data.zero_()  # 清空梯度
    print('Predict(after training)', 4, forward(4).item())