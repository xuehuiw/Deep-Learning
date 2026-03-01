#07 Dataset and DataLoader
##在深度学习中，Dataset和DataLoader是PyTorch中用于处理和加载数据的两个重要类。Dataset是一个抽象类，用户需要继承这个类并实现其中的__init__、__getitem__和__len__方法，以便创建自己的数据集类。DataLoader则是一个工具类，用于将Dataset中的数据加载到模型中进行训练或评估。
##Dataset类的__init__方法用于初始化数据集对象，通常会加载数据、进行预处理等操作。__getitem__方法用于通过索引访问数据集中的每个样本，返回对应的输入特征和输出特征。__len__方法返回数据集的长度，即样本的数量。这使得DataLoader能够知道数据集中有多少个样本，以便在训练过程中正确地迭代数据。
##DataLoader类提供了批量加载、打乱数据和多线程加载等功能。通过将自定义的数据集类传递给DataLoader，我们可以轻松地在训练过程中迭代数据，并且可以指定批量大小、是否打乱数据以及使用多少个线程来加载数据。这使得训练过程更加高效和灵活，特别是在处理大型数据集时。
##准备数据集、设计模型、构造损失函数和优化器、训练模型

###以下是对糖尿病数据集进行处理和训练的示例代码：
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):#dataset是一个抽象类，无法进行实例化，用户需要继承这个类并实现其中的__init__、__getitem__和__len__方法，以便创建自己的数据集类。
    def __init__(self, file_path):#__init__方法是数据集类的构造函数，用于初始化数据集对象。在这个方法中，通常会加载数据、进行预处理等操作。这里的file_path参数是数据文件的路径，用户可以根据自己的需要进行修改。
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)#使用numpy的loadtxt函数加载数据文件，指定分隔符为逗号，并将数据类型设置为float32。加载后的数据存储在变量xy中。
        self.len = xy.shape[0]  # 样本数量
        self.x_data = torch.from_numpy(xy[:, :-1])  # 输入特征取前面所有列，转换为PyTorch张量
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 输出特征取最后一列，转换为PyTorch张量

    def __getitem__(self, index):#index是样本的索引，返回对应的输入特征和输出特征。这个方法使得数据集能够通过索引访问每个样本的数据。
        return self.x_data[index], self.y_data[index]

    def __len__(self):#返回数据集的长度，即样本的数量。这使得数据加载器能够知道数据集中有多少个样本，以便在训练过程中正确地迭代数据。
        return self.len
    
dataset = DiabetesDataset('C:/Users/18261/Desktop/Deep-Learning/刘二深度学习/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
# dataloader是PyTorch中用于加载数据的工具，可实例化。它提供了批量加载、打乱数据和多线程加载等功能。
# 通过将自定义的数据集类传递给DataLoader，我们可以轻松地在训练过程中迭代数据，并且可以指定批量大小、是否打乱数据以及使用多少个线程来加载数据。

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8 , 6)  # 输入维度为8，输出维度为6
        self.linear2 = torch.nn.Linear(6 , 4)  # 输入维度为6，输出维度为4
        self.linear3 = torch.nn.Linear(4 , 1)  # 输入维度为4，输出维度为1
        self.activate = torch.nn.Sigmoid()  # 激活函数

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x
    
model = Model()

criterion = torch.nn.BCELoss(reduction='mean')  # 均方误差损失函数，
#6、7节的损失函数不同，6节使用的是MSELoss（均方误差损失函数），适用于回归问题（数值）；而7节使用的是BCELoss（二元交叉熵损失函数），适用于二分类问题（概率）。根据具体的任务需求选择合适的损失函数是非常重要的。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，model.parameters()返回模型的所有可学习参数，lr=0.01指定学习率为0.01。优化器将根据计算得到的损失值来更新模型的参数，以便在训练过程中逐步优化模型的性能。

loss_list = []  # 用于记录每个epoch的loss

for epoch in range(100):#epoch是训练的轮数，表示整个数据集被模型完整地训练一次。在每个epoch中，模型会迭代整个数据集，并根据损失函数计算损失值，然后通过反向传播更新模型参数。通常，训练过程中会监控每个epoch的损失值，以评估模型的性能和收敛情况。
    epoch_loss = 0  # 记录本轮总loss
    batch_count = 0
    for i, data in enumerate(train_loader, 0):#0表示从第0个批次开始迭代，i是批次的索引，data是当前批次的数据。通过enumerate(train_loader)可以同时获取批次的索引和数据，方便在训练过程中进行监控和调试。
        inputs, labels = data# loader会直接把每个data转化成tensor的形式，inputs是输入特征，labels是对应的标签。
        #=> i,(inputs, labels) i,(x_data, y_data) 
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print('Epoch:', epoch, 'Batch:', i, 'Loss:', loss.item())
        optimizer.zero_grad()#在进行反向传播之前，通常需要先将优化器的梯度缓存清零，以避免梯度累积。optimizer.zero_grad()就是用来清除之前计算的梯度值，确保每次反向传播时只计算当前批次的梯度。
        loss.backward()#loss.backward()是PyTorch中用于执行反向传播的函数。它会计算损失函数相对于模型参数的梯度，并将这些梯度存储在每个参数的.grad属性中。
        optimizer.step()#optimizer.step()是PyTorch中用于更新模型参数的函数。它会根据之前计算的梯度值来调整模型的参数，以最小化损失函数。
        epoch_loss += loss.item()
        batch_count += 1
    loss_list.append(epoch_loss / batch_count)  # 记录每个epoch的平均loss
    print(f'Epoch: {epoch}, Average Loss: {epoch_loss / batch_count}')
# 在这个训练循环中，我们首先迭代每个epoch，然后在每个epoch中迭代每个批次的数据。对于每个批次，我们将输入数据传递给模型，计算预测结果y_pred，并使用损失函数criterion计算损失。然后，我们进行反向传播并更新模型参数。通过这种方式，我们可以有效地训练模型，并且可以监控每个epoch和每个批次的损失值。
# 然后，我们将输入数据传递给模型，计算预测结果y_pred，并使用损失函数criterion计算损失。
# 最后，我们进行反向传播并更新模型参数。通过这种方式，我们可以有效地训练模型，并且可以监控每个epoch和每个批次的损失值。

# 绘制loss曲线
plt.figure()
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

#6、7节在数据准备方面的区别，6节直接使用numpy加载数据并转换为PyTorch张量，而7节则通过定义一个自定义的Dataset类来加载数据，并使用DataLoader进行批量加载。这种方式更适合处理大型数据集，并且可以更灵活地进行数据预处理和增强。
#在训练模型方面的区别，6节直接在一个循环中进行训练，而7节则使用了嵌套的循环结构，外层循环迭代每个epoch，内层循环迭代每个批次的数据。这种方式更适合处理大型数据集，并且可以更好地监控每个epoch和每个批次的损失值。