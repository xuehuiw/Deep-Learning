#随机梯度下降算法SGD
##每轮对每个样本计算梯度（计算单个样本的梯度），更新权重，更新不稳定（噪声大、波动明显）但每轮计算量小、更新更快
#小批量梯度下降算法MBGD（Mini-batch）
##每轮对一些小批量样本计算梯度（计算小批量样本的平均梯度），更新权重，更新稳定（噪声较小、波动较小）且每轮计算量适中、更新较快

x_data= [1.0, 2.0, 3.0]
y_data= [2.0, 4.0, 6.0]

w=1.0

def forward(x):
    return x*w

def loss(x,y):
        y_pred = forward(x)
        return(y_pred-y)**2

def gradient(x,y):
        y_pred = forward(x)
        return 2*x*(y_pred-y)

print('Predict(before training)', 4, forward(4))
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad_val = gradient(x_val, y_val)
        w -= 0.01*grad_val
        print("\tgrad:",x_val, y_val, grad_val)
        l=loss(x_val, y_val)
    print('Epoch:', epoch, 'w=', w, 'loss=', l)
print('Predict(after training)', 4, forward(4))