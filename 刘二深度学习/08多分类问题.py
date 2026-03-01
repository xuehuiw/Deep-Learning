#08 Softmax Classifier
##Softmax函数可以将模型的输出转换为概率分布，使得每个类别的预测值都在0到1之间，并且所有类别的预测值之和为1。
##NLLLoss（Negative Log Likelihood Loss）是PyTorch中用于多分类问题的损失函数。NLLLoss接受模型输出的对数概率（log probabilities）作为输入，并计算负对数似然损失。通过最小化这个损失，模型可以学习到更准确的分类结果。
###NLLLoss+Softmax的组合通常被称为CrossEntropyLoss（交叉熵损失函数），它在内部同时计算Softmax和NLLLoss。这种组合使得我们在训练多分类模型时更加方便和高效，因为我们只需要使用一个损失函数来处理整个过程，而不需要分别计算Softmax和NLLLoss。
##在处理多分类问题时，逻辑回归模型的输入特征可以是一个向量或矩阵。对于每个样本，输入特征可以包含多个维度，例如多个数值特征、类别特征等。逻辑回归模型会将这些多维输入特征进行线性组合，并通过逻辑函数将结果映射到一个概率值。
##在训练过程中，逻辑回归模型会学习每个输入特征的权重，以便更好地拟合数据。对于多维输入，模型会考虑每个特征对预测结果的贡献，并通过优化算法来调整权重，使得模型能够更准确地预测类别标签。
##在实际应用中，处理多维输入时可能需要进行特征工程，例如对类别特征进行独热编码（one-hot encoding）或使用嵌入层（embedding layer）来表示高维稀疏特征。此外，正则化技术（如L1或L2正则化）也可以用于防止过拟合，特别是在处理高维输入时。
##总之，逻辑回归模型能够处理多维输入特征，并通过学习权重来进行分类预测。正确地处理和预处理多维输入特征对于模型的性能和泛化能力至关重要。
'''
#以下是交叉熵损失函数的使用示例代码：
import torch
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
Y = torch.LongTensor([0, 1, 2])  # 真实标签
Y_pred1 = torch.FloatTensor([[0.5, 0.1, 0.4],  # 预测值（未经过Softmax）
                              [0.3, 0.6, 0.1],
                              [0.1, 0.1, 0.8]])
Y_pred2 = torch.FloatTensor([[0.2, 0.7, 0.1],  # 预测值（未经过Softmax）
                              [0.1, 0.3, 0.6],
                              [0.2, 0.5, 0.3]])

l1 = criterion(Y_pred1, Y)
l2 = criterion(Y_pred2, Y)
print('Loss1:', l1.data)
print('Loss2:', l2.data)
'''
#接下来是一个手写数字识别的示例代码：
##图片是一个28x28的灰度图像，输入层有784个神经元（28*28=784），输出层有10个神经元（对应数字0-9）。
##通道数为1，因为是灰度图像。
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
'''
1. torch.utils.data.Dataset
- 是PyTorch里最基础的数据集“模板”。
- 你要用自己的数据时，需要继承它，自己写怎么加载数据、怎么取样本。
- 适合自定义各种奇怪的数据格式。

2. torchvision.datasets
- 是官方帮你写好的“现成数据集”，比如MNIST、CIFAR10、ImageNet等。
- 直接用，不用自己写加载和处理，常见图片数据集都能一行代码搞定。
- 其实它们内部也是继承了torch.utils.data.Dataset，只是帮你封装好了。

一句话总结：
- Dataset 是你自己写数据集的“模板”。
- torchvision.datasets 是官方帮你写好的“现成数据集”，直接用就行。
'''
import torch.nn.functional as F
import torch.optim as optim

#Step1:准备数据集
batch_size = 64
transforms = transforms.Compose([
    transforms.ToTensor(), #将PIL Image转换为torch.FloatTensor，并且归一化到[0.0, 1.0]之间。本来是28*28的图像，经过ToTensor后变成了1*28*28的张量。pixel值从0到255被归一化到0-1之间。
    transforms.Normalize((0.1307,), (0.3081,))#对图像进行标准化处理，使用MNIST数据集的均值0.1307和标准差0.3081进行归一化。（这两个值由MNIST数据集的统计特征计算得出，确保输入数据的分布更适合模型训练。）这个操作可以帮助模型更快地收敛，提高训练效果。
    ])                     #mean       std

train_dataset = datasets.MNIST(root='C:\\Users\\18261\\Desktop\\Deep-Learning\\刘二深度学习\\dataset', 
                               train=True, 
                               download=True, 
                               transform=transforms)
train_loader = DataLoader(train_dataset,
                           batch_size=batch_size,
                             shuffle=True)
test_dataset = datasets.MNIST(root='C:\\Users\\18261\\Desktop\\Deep-Learning\\刘二深度学习\\dataset', 
                              train=False, #test数据集无需训练
                              download=True, 
                              transform=transforms)
test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=False)#测试数据集不需要打乱顺序，因为我们通常在评估模型性能时希望保持数据的原始顺序，以便更准确地计算指标。

#Step2:设计模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512) #输入层到隐藏层的全连接层，输入维度为28*28=784，输出维度为128。
        self.l2 = torch.nn.Linear(512, 256)   
        self.l3 = torch.nn.Linear(256, 128)     
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10) 

    def forward(self, x):
        x = x.view(-1, 784)  #将输入图像展平为一维向量，-1表示自动推断维度。
        '''
        -1在view()里的用法：
        1.把图像展平为一维向量
import torch
# 假设是一批 MNIST 图像：4张，单通道，28x28
x = torch.randn(4, 1, 28, 28)
print(x.shape)  # torch.Size([4, 1, 28, 28])
# 展平成 [batch_size, feature_dim]
y = x.view(-1, 784)
print(y.shape)  # torch.Size([4, 784])
        2.把矩阵重塑为向量
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape: (2, 3)
b = a.view(-1)
print(b.shape)  # torch.Size([6])
print(b)        # tensor([1, 2, 3, 4, 5, 6])
        3.重塑为三维张量
c = torch.randn(24)  # shape: (24)
d = c.view(2, -1, 3)
print(d.shape)  # torch.Size([2, 4, 3])
'''
        x = F.relu(self.l1(x))#使用ReLU激活函数对第一层的输出进行非线性变换。
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)        #第五层的输出，即最终的分类结果（未经过Softmax）。
        return x

model = Net()

#Step3:构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失函数，适用于多分类问题。
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.5)  #随机梯度
#momentum参数可以帮助优化器在更新参数时考虑之前的更新方向，从而加速收敛并减少震荡。通过设置momentum=0.5，我们可以让优化器在每次更新时考虑之前更新的一半，这有助于提高训练效率和模型性能。

#Step4:训练模型
def train(epoch):#这里定义了一个train函数，将训练过程封装起来，方便在每个epoch中调用。
    running_loss = 0.0#running_loss用于累积每个批次的损失值，以便在每个epoch结束时计算平均损失。
    for batch_idx, data in enumerate(train_loader,0):
        #batch_idx是批次的索引，data是当前批次的输入数据，target是当前批次的标签,0表示从第0个批次开始迭代。通过enumerate(train_loader)可以同时获取批次的索引和数据，方便在训练过程中进行监控和调试。
        inputs, target = data
        optimizer.zero_grad()  #在进行反向传播之前，通常需要先将优化器的梯度缓存清零，以避免梯度累积。optimizer.zero_grad()就是用来清除之前计算的梯度值，确保每次反向传播时只计算当前批次的梯度。
        output = model(inputs)   #将输入数据传递给模型，得到预测结果output。
        loss = criterion(output, target)  #使用损失函数criterion计算预测结果output与真实标签target之间的损失值。
        loss.backward()        #执行反向传播，计算损失函数相对于模型参数的梯度，并将这些梯度存储在每个参数的.grad属性中。
        optimizer.step()       #根据之前计算的梯度值来调整模型的参数，以最小化损失函数。

        running_loss += loss.item()  #将当前批次的损失值累积到running_loss中。
        if batch_idx % 300 == 299:    #每300个批次打印
            print('[Epoch: %d, Batch: %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0  #重置running_loss为0，以便在下一个300个批次中重新累积损失值。

def test():#这里定义了一个test函数，将测试过程封装起来，用于评估模型在测试集上的性能。
    correct = 0#correct用于统计模型在测试集上正确预测的样本数量。
    total = 0#total用于统计测试集中的总样本数量，以便在最后计算准确率。
    with torch.no_grad():#在评估模型时，我们通常不需要计算梯度，因此使用torch.no_grad()上下文管理器来禁用梯度计算。这可以节省内存和计算资源，提高评估效率。
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) #torch.max函数返回每行的最大值和对应的索引，这里我们只需要索引（即预测的类别标签），_是用来接收最大值的占位符，因为我们不需要它。
            #dim=1表示我们要在每行中找到最大值，即在每个样本的输出中找到预测概率最高的类别。
            total += labels.size(0) #labels.size(0)中0表示我们要获取第0维的大小，即批次中的样本数量。返回当前批次中样本的数量，将其累积到total中，以便在最后计算准确率。
            correct += (predicted == labels).sum().item() #predicted == labels会返回一个布尔张量，表示每个预测是否正确。通过.sum()计算正确预测的数量，并将其累积到correct中。

    print('Accuracy on test set: %d %%' % (
        100 * correct / total))#最后，计算并打印模型在测试集上的准确率，即正确预测的数量除以总样本数量，并乘以100得到百分比形式。
    
if __name__ == '__main__':
    for epoch in range(10):  #迭代10个epoch进行训练。
        train(epoch)         #在每个epoch中调用train函数进行训练。
        test()               #在每个epoch结束后调用test函数评估模型性能。



'''
【面试问答总结：手写数字识别项目】

问题1：你的模型有5层全连接层，为什么选择这样的结构？能否解释每一层的作用？
解答：采用多层全连接层是为了让模型在“线性变换+非线性激活”的交替结构中，逐步提取和组合输入图像的层级特征（逐步评估局部到全局特征的重要程度）。
选用5层以及特定的神经元下降梯度（784->512->256->128->64->10）是基于实验调参得到的较优结构。

问题2：代码中用到了 transforms.Normalize((0.1307,), (0.3081,))，这两个值是什么含义？
解答：这两个值分别是 MNIST 数据集所有图像像素的全局均值（0.1307）和标准差（0.3081）。这步操作是为了将输入数据标准化为均值为0、方差为1的状态。
这样可以使得神经网络在训练时各个维度的特征尺度一致，从而极大加速模型收敛并提升训练效果。

问题3：你用的损失函数是 CrossEntropyLoss，能解释为什么分类问题不用 MSELoss（均方误差）吗？
解答：MSE 主要用于回归问题。在分类问题中，网络末端通常有 Softmax 或 Sigmoid 激活函数输出概率。如果配合 MSE 使用，当模型预测极端错误时，
激活函数的导数会趋近于0，引发“梯度消失”（错得越离谱越不更新参数）。而交叉熵（CrossEntropy）引入了对数运算，其求导结果在数学上刚好抵消了激活函数的导数，
使得误差越大，计算出的梯度也就是惩罚力度越大，模型修正得越快。

问题4：优化器中的 momentum=0.5 是什么意思？
解答：Momentum（动量）是帮助优化器在当前的梯度更新中，保留一部分上一轮更新的方向和幅度（0.5代表保留之前的一半）。这类似物理里的惯性，
能够帮助模型在平坦的损失曲面上加速收敛，减少震荡，并有助于冲出局部最优解。

问题5：为什么要在 optimizer.step() 之前调用 optimizer.zero_grad()？如果不清零会发生什么？
解答：PyTorch 的底层机制默认是“梯度累加”而不是覆盖。如果不清零，当前 batch 算出的梯度会和之前所有 batch 的梯度加在一起，导致更新方向完全混乱。
因此在每次反向传播（loss.backward()）前，必须手动清零，保证优化器只用当前轮次的梯度去更新参数。

问题6：代码中用准确率（accuracy）来评估模型，对于不平衡数据集，这个指标够吗？还可以用什么指标？
解答：不够。在严重不平衡的数据集（比如99%正样本，1%负样本）里，模型只要无脑全猜正样本，准确率就有99%，但这没有实际意义。此时还需要引入这几个指标：
1. 召回率（Recall）：实际为正的样本中，模型找出了多少（宁可错杀一千绝不放过一个，适合疾病筛查）。
2. 精确率（Precision）：模型预测为正的样本中，真正为正的比例有几成。
3. F1-score：精确率和召回率的调和平均值，用于综合评估。
'''

