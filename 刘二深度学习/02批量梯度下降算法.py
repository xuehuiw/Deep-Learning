#02 Gradient Descent
##批量梯度下降算法BGD
###是一个优化算法，用于训练机器学习模型，特别是线性回归和神经网络等。它通过计算整个训练数据集的梯度来更新模型的参数，以最小化损失函数。
###每轮用全部样本进行计算一次梯度（平均值），更新权重，更新稳定但每轮计算量大

import matplotlib.pyplot as plt
import numpy as np

x_data= [1.0, 2.0, 3.0]
y_data= [2.0, 4.0, 6.0]

w=1.0

def forward(x):
    return x*w

def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred-y)**2
    return cost/len(xs)

def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        grad += 2*x*(y_pred-y)
    return grad/len(xs)

epoch_list = []
cost_list = []
w_list = []
print('Predict(before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)  # 把当前损失存起来
    epoch_list.append(epoch)
    w_list.append(w)  # 把当前w存起来
    grad_val = gradient(x_data, y_data)
    w -= 0.01*grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('Predict(after training)', 4, forward(4))

# 绘制双Y轴图
fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建主画布和第一个轴（左Y轴）

# -------------- 第一个Y轴：损失（Cost）--------------
color1 = '#1f77b4'  # 蓝色
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Cost', color=color1, fontsize=12)
ax1.plot(range(100), cost_list, color=color1, linewidth=2, label='Cost')
ax1.tick_params(axis='y', labelcolor=color1)  # 左Y轴刻度文字设为蓝色
ax1.grid(True, alpha=0.3)

# -------------- 第二个Y轴：w值 --------------
ax2 = ax1.twinx()  # 共享X轴，创建第二个Y轴（右Y轴）
color2 = '#d62728'  # 红色
ax2.set_ylabel('w', color=color2, fontsize=12)
ax2.plot(range(100), w_list, color=color2, linewidth=2, label='w')
ax2.tick_params(axis='y', labelcolor=color2)  # 右Y轴刻度文字设为红色

# -------------- 美化和标注 --------------
fig.tight_layout()  # 防止标签重叠
plt.title('Cost and w Change with Epochs', fontsize=14)

# 添加图例（同时显示两个指标）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.show()