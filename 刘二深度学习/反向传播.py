#03 Back Propagation
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