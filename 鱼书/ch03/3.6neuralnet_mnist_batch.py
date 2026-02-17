import sys, os

# ...existing code...
# 替换原来的 sys.path.append(os.pardir)
# 使用 __file__ 获取当前文件路径，更加稳健，不管你在哪里运行都不会错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    # 修改：获取当前脚本文件所在的目录 (ch03)，而不是上一级目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "sample_weight.pkl")
    
    print(f"正在尝试加载权重文件: {file_path}")
    
    if not os.path.exists(file_path):
        print("\n❌ 错误：找不到 sample_weight.pkl 文件！")
        print(f"请确认该文件是否在文件夹: {current_dir} 中")
        # 抛出异常终止程序，防止后续报错
        raise FileNotFoundError("缺少权重文件 sample_weight.pkl")

    with open(file_path, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
