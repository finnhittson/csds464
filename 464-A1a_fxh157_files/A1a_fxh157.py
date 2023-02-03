import math
import matplotlib.pyplot as plt
import numpy as np


# Returns value on normal distribution
def g(x, mu: float=0.0, sigma: float=1.0):
    return 1 / math.sqrt(2*math.pi*sigma**2) * math.exp(-(x-mu)**2 / (2*sigma**2))


# Plots normal distribution and highlights specific value x1
def plot_normal(mu: float=0.0, sigma: float=1.0, x1: float=None):
    # points along x and y axis
    x = np.linspace(-4*sigma+mu, 4*sigma+mu, 100)
    y = [g(point, mu=mu, sigma=sigma) for point in x]
    plt.plot(x,y)

    # plots passed specific value
    if x1 is not None:
        # line from x-axis to curve
        plt.plot([x1, x1], [0, g(x1, mu=mu, sigma=sigma)], 'r')
        # single point on curve
        plt.scatter(x1, g(x1, mu=mu, sigma=sigma), s=40, c='r', zorder=10)
    
    # formating of point label
    pad = 0.1
    plt.annotate(f"$p(x1={x1}|\\mu,\\sigma)$={round(g(x1, mu=mu, sigma=sigma),3)}", (x1+pad, g(x1, mu=mu, sigma=sigma)))

    # axes labels and title formatting
    plt.margins(y=0)
    plt.margins(x=0)
    plt.ylim([0,g(mu,mu=mu,sigma=sigma)+0.05])
    plt.xlabel("x", fontsize=16)
    plt.ylabel("$p(x|\\mu,\\sigma)$", fontsize=16)
    plt.title(f"Normal Probability Density Function, $\\mu={mu}$, $\\sigma={sigma}$", fontsize=18)
    plt.show()

plot_normal(mu=-1, sigma=0.5, x1=-2)