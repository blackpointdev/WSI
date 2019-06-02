import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def obj_func(x):
    w1 = 1 + (x[0]-1)/4
    w2 = 1 + (x[1] - 1) / 4

    sum = (w1 - 1)**2 * (1 + 10 * math.sin(math.pi * w1 + 1)**2)

    return math.sin(math.pi * w1)**2 + sum + (w2 - 1)**2 * (1 + math.sin(2 * math.pi * w2)**2)

def draw_function():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    z = np.array([obj_func([x_z, y_z]) for x_z, y_z in zip(np.ravel(x), np.ravel(y))])
    z = z.reshape(x.shape)
    surf = ax.plot_surface(x, y, z, cmap='inferno', linewidth=0)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Levy function (x, y)')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

draw_function()