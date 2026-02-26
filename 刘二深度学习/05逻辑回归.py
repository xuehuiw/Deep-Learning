#05 Logistic Regression
##逻辑回归是一种用于分类问题的监督学习算法，特别适用于二分类问题。它通过使用一个逻辑函数（如sigmoid函数）将线性组合的输入特征映射到一个概率值，从而进行分类。
##逻辑回归模型假设目标变量与特征变量之间存在线性关系，但通过逻辑函数将线性组合的结果转换为概率值。模型的输出是一个介于0和1之间的概率值，表示输入数据属于某个类别的概率。
##逻辑回归的目标是找到一组参数，使得模型预测的概率值与实际的类别标签之间的误差最小化。常用的误差度量是对数损失（log loss），它计算预测概率与实际类别标签之间的平均对数差。
##逻辑回归模型可以通过最大似然估计来求解参数，或者使用梯度下降等优化算法来迭代更新参数。逻辑回归不仅可以用于二分类问题，还可以通过扩展到多分类问题（如softmax回归）来处理多类别分类任务。
##他和线性回归的区别在于，线性回归用于预测连续的数值，而逻辑回归用于预测离散的类别标签。逻辑回归通过使用逻辑函数将线性组合的输入特征映射到一个概率值，从而进行分类，而线性回归直接输出一个连续的数值。
import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))  # 使用sigmoid函数将线性输出转换为概率值
        return y_pred
    
model= LogisticRegressionModel()
criterion = torch.nn.BCELoss(size_average=False)  # 二分类交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch:', epoch, 'Loss:', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view(200, 1)
y_t = model(x_t)
y = y_t.data.numpy()
plt.figure()
plt.plot(x, y)
plt.plot([0,10], [0.5, 0.5], c = 'r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.title('Logistic Regression')
plt.grid()
plt.show()

#4,5节网络模型框架结构编写：
# 准备数据（复杂时可以提前封装）读取数据；Prepare dataset, dataloader等；
# 模型构造（定义模型，复杂时可单独设置文件然后在主文件里import）；Design model using Class
# 损失函数和优化器构造；Construct loss and optimizer
# 训练循环（前向传播，计算损失，反向传播，更新参数）；training cycle
# 评估模型（测试数据预测，计算准确率等指标）。
##后续可以根据需求扩充各个部分