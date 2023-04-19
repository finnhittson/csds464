import matplotlib.pyplot as plt
import numpy as np

def plot_circle(N, k, output:bool=False):
    ns = np.array(list(range(0, N+1)))
    x = [np.cos(2*np.pi*k/N*n) for n in ns]
    y = [np.sin(2*np.pi*k/N*n) for n in ns]
    
    plt.figure().set_figheight(5)
    plt.figure().set_figwidth(5)
    plt.plot(x, y)
    plt.scatter(x, y, c='r', s=5, zorder=10)
    plt.title(f"unit circle, $N={N}$, $k={k}$")
    plt.show()

    if output:
        return x, y