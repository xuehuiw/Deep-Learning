import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 6, 8, 10]

def forward(x):
    return w * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_list = []
b_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 4.1, 0.1):
        print("w=", w, "b=", b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print("\t", x_val, y_val, y_pred_val, loss_val)
        print("MSE=", l_sum / 5)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(w_list, b_list, mse_list, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()