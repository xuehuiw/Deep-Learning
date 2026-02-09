import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return (x> 0).astype(np.int32)

x= np.arange(-5.0, 5.0, 0.1)
y= step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()