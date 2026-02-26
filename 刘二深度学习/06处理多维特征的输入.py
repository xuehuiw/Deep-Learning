#06 Multiple Dimension Input
##在处理多维输入时，逻辑回归模型的输入特征可以是一个向量或矩阵。对于每个样本，输入特征可以包含多个维度，例如多个数值特征、类别特征等。逻辑回归模型会将这些多维输入特征进行线性组合，并通过逻辑函数将结果映射到一个概率值。
##在训练过程中，逻辑回归模型会学习每个输入特征的权重，以便更好地拟合数据。对于多维输入，模型会考虑每个特征对预测结果的贡献，并通过优化算法来调整权重，使得模型能够更准确地预测类别标签。
##在实际应用中，处理多维输入时可能需要进行特征工程，例如对类别特征进行独热编码（one-hot encoding）或使用嵌入层（embedding layer）来表示高维稀疏特征。此外，正则化技术（如L1或L2正则化）也可以用于防止过拟合，特别是在处理高维输入时。
##总之，逻辑回归模型能够处理多维输入特征，并通过学习权重来进行分类预测。正确地处理和预处理多维输入特征对于模型的性能和泛化能力至关重要。

import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('C:/Users/18261/Desktop/Deep-Learning/刘二深度学习/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 输入特征
y_data = torch.from_numpy(xy[:, [-1]])  # 输出特征

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8 , 6)  # 输入维度为8，输出维度为6
        self.linear2 = torch.nn.Linear(6 , 4)  # 输入维度为6，输出维度为4
        self.linear3 = torch.nn.Linear(4 , 1)  # 输入维度为4，输出维度为1
        self.activate = torch.nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x
    
model = Model()
criterion = torch.nn.MSELoss(size_average=True)  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
loss_list = []  # 用于记录每个epoch的loss

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch:', epoch, 'Loss:', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())  # 记录loss

# 绘制loss曲线
plt.figure()
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()