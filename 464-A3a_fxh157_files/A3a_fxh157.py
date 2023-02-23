import math
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../464-A1b_fxh157_files/')
import A1b_fxh157 as a1b

# 1. Continuous signals and sampling
## 1a. Sampled functions
def plot_sampled_function(g, fs:int=1, tlim:tuple=None, tscale:float=1.0, tunits:str="sec", **kwargs):
    t = np.arange(tlim[0], tlim[1], tlim[1]/(100000))
    y = g(t=t, **kwargs)
    plt.plot(t, y)
    norm = 1
    if g is a1b.gammatone:
        norm = a1b.gammatone_norm(y)

    sample_times = np.arange(tlim[0], tlim[1], tscale/fs)
    if g is a1b.sinewave:
        sample_times = np.arange(tlim[0], tlim[1], 1/fs)*tscale
    for st in sample_times:
        y = g(t=st, **kwargs)/norm
        plt.plot([st, st], [0, y], 'r')
        plt.scatter(st, y, c='r', s=15, zorder=10)
    
    plt.xlim([t[0], t[-1]])
    plt.xlabel(f"time $t$, ({tunits})", fontsize=16)
    plt.ylabel("amplitude", fontsize=16)

# 2. Signals
## 2a. Delta and step functions
def d(t, fs:int=1):
    return [1 if round(i, 3) == 0 else 0 for i in t]
    
def u(t):
    return [1 if i >= 0 else 0 for i in t]

def plot_delta_step(t, fs, g, plot_type:str="line"):
    t = np.arange(-t, t+1, 1/fs)
    if g == d:
        y = g(t, fs)
    else:
        y = g(t)
    if plot_type == "stem":
        for i in range(len(y)):
            plt.plot([t[i], t[i]], [0, y[i]], '#1f77b4')
            plt.scatter(t[i], y[i], c='#1f77b4', s=15, zorder=10)
    else:
        plt.plot(t,y)
    plt.ylim([0,1.5])
    plt.yticks([1])
    plt.show()

## 2b. gensignal


# 3. Noise and SNR
## 3a. energy, power, and snr
def energy(x):
    return np.linalg.norm(x)**2

def power(x):
    return np.linalg.norm(x)**2/len(x)

def snr(Ps, Pn):
    return Ps/Pn