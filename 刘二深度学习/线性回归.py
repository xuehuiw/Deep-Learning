#04 Linear Regression
##线性回归是一种基本的监督学习算法，用于预测一个连续的目标变量（也称为因变量）与一个或多个特征变量（也称为自变量）之间的关系。
##线性回归模型假设目标变量与特征变量之间存在线性关系，即目标变量可以表示为特征变量的线性组合。
###线性回归的目标是找到一组参数，使得模型预测的目标变量与实际的目标变量之间的误差最小化。常用的误差度量是均方误差（MSE），它计算预测值与实际值之间的平均平方差。
###线性回归模型可以通过最小二乘法来求解参数，或者使用梯度下降等优化算法来迭代更新参数。
import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):# 定义模型的结构
        super(LinearModel, self).__init__() #调用父类的构造函数
        self.linear = torch.nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):# 定义前向传播的计算过程
        y_pred = self.linear(x)# 线性变换，计算预测值
        return y_pred# 返回预测值
    
model = LinearModel()# 创建模型实例
criterion = torch.nn.MSELoss(reduction='sum')  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器


# 保存每个epoch的loss用于画图
loss_list = []
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss_list.append(loss.item())
    print('Epoch:', epoch, 'Loss:', loss.item())
    optimizer.zero_grad()#清空之前的梯度
    loss.backward()#反向传播计算当前梯度
    optimizer.step()#更新参数

# 绘制loss曲线
plt.figure()
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

print('w=', model.linear.weight.item(), 'b=', model.linear.bias.item())

# Test the model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('Predict (after training)', y_test.data)