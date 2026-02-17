class MulLayer:
    def __init__(self):#初始化成员变量,在前向传播时用于保存输入的值,在反向传播时使用
        self.x = None
        self.y = None

    def forward(self, x, y):#前向传播,相乘输出并保存输入的值
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):#反向传播,计算梯度,根据链式法则,输出的梯度dout乘以输入的值x和y分别得到dx和dy
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

apple=100
apple_num=2
tax=1.1

#创建乘法层对象layer
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()

#前向传播forward
apple_price=mul_apple_layer.forward(apple, apple_num) #计算苹果的总价钱
price=mul_tax_layer.forward(apple_price, tax) #计算含税价格

print('价格:', price)

#反向传播backward
dprice=1.0 #价格的梯度
dapple_price, dtax = mul_tax_layer.backward(dprice) #计算含税价格对苹果总价钱和税的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #计算苹果总价钱对苹果单价和数量的梯度

print('苹果总价的梯度:', dapple_price)
print('苹果单价的梯度:', dapple)
print('苹果数量的梯度:', dapple_num)
print('税的梯度:', dtax)

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1

#layer
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()

#forward
apple_price=mul_apple_layer.forward(apple, apple_num) #计算苹果的总价钱
orange_price=mul_orange_layer.forward(orange, orange_num) #计算橘子的总价钱
all_price=add_apple_orange_layer.forward(apple_price, orange_price) #计算总价钱
price=mul_tax_layer.forward(all_price, tax) #计算含税价格

#backward
dprice=1.0 #价格的梯度
dall_price, dtax = mul_tax_layer.backward(dprice) #计算含税价格对总价钱和税的梯度
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #计算总价钱对苹果总价钱和橘子总价钱的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #计算苹果总价钱对苹果单价和数量的梯度
dorange, dorange_num = mul_orange_layer.backward(dorange_price) #计算橘子总价钱对橘子单价和数量的梯度

print('苹果单价的梯度:', dapple)
print('苹果数量的梯度:', dapple_num)
print('橘子单价的梯度:', dorange)
print('橘子数量的梯度:', dorange_num)
print('税的梯度:', dtax)
