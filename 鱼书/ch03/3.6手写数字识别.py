import sys, os

# ...existing code...
# 替换原来的 sys.path.append(os.pardir)
# 使用 __file__ 获取当前文件路径，更加稳健，不管你在哪里运行都不会错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# ...existing code...
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)