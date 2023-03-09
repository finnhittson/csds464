import math
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('../464-A1b_fxh157_files/')
import A1b_fxh157 as a1b

# 1. Continuous signals and sampling
## 1a. Sampled functions
def plot_sampled_function(g, fs:int=1, tlim:tuple=None, tscale:float=1.0, tunits:str="sec", title:str="set me", **kwargs):
    t = np.arange(tlim[0], tlim[1], tlim[1]/(100000))*tscale
    y = g(t=t, **kwargs)
    plt.plot(t, y)

    norm = 1
    if g is a1b.gammatone:
        norm = a1b.gammatone_norm(y)
    
    sample_times = np.arange(tlim[0], tlim[1], 1/fs)
    if g is a1b.sinewave:
        sample_times = np.arange(tlim[0]*tscale, tlim[1]*tscale, 1/fs)
    
    for st in sample_times:
        y = g(t=st, **kwargs)/norm
        plt.plot([st, st], [0, y], 'r')
        plt.scatter(st, y, c='r', s=15, zorder=10)
    
    plt.xlim([t[0], t[-1]])
    if tunits == "msec":
        ticks = plt.xticks()[0]
        plt.xticks(ticks=ticks, labels=ticks*1000)
    plt.xlabel(f"time $t$, ({tunits})", fontsize=16)
    plt.ylabel("amplitude", fontsize=16)
    plt.title(title, fontsize=18)
    if 'f' in kwargs:
        plt.title(title + f", ${kwargs['f']}Hz$", fontsize=18)
    plt.show()

# 2. Signals
## 2a. Delta and step functions
def d(t):
    if isinstance(t, int) or isinstance(t, float):
        return 1 if t == 0 else 0
    return np.array([1 if round(i, 3) == 0 else 0 for i in t])
    
def u(t):
    return 1 if t >= 0 else 0

def plot_delta_step(t0, tn, g, fs:int=1, tau:float=0.0, T:float=0.0, title:str="set me", tscale:float=1.0, tunits:str="sec", plot_type:str="line"):
    fs *= tscale
    t = np.arange(t0, tau+T if g!=d else tn, 1/fs)
    y = [g(i-tau) for i in t]
    if tn > tau+T:
        ta = np.arange(t[-1]+1/fs, tn, 1/fs)
        t = np.concatenate((t, ta))
        y = np.concatenate((y, np.zeros(len(ta))))
    if plot_type.lower() == "stem":
        for i in range(len(y)):
            plt.plot([t[i], t[i]], [0, y[i]], '#1f77b4')
            plt.scatter(t[i], y[i], c='#1f77b4', s=15, zorder=10)
    elif plot_type.lower() == "line":
        plt.plot(t,y)
    else:
        print("plot type not supported.")
        return
    plt.xlabel(f"time, ({tunits})", fontsize=16)
    plt.title(f"{title}, $\\tau={tau}\\ {tunits}$, $T={T}\\ {tunits}$", fontsize=18)
    plt.ylim([0,1.5])
    plt.yticks([1])
    plt.show()

## 2b. gensignal
def gensignal(t0, tn, g, fs:int=1, tau:float=0.0, T:float=0.0, tscale:float=1, **kwargs):
    fs = fs*tscale
    t = []
    y = []
    if g == u:
        t = np.arange(t0, tau+T, 1/fs)
        y = [g(i-tau) for i in t]
        if tn > tau+T:
            ta = np.arange(t[-1]+1/fs, tn, 1/fs)
            t = np.concatenate((t, ta))
            y = np.concatenate((y, np.zeros(len(ta))))
    elif g == d:
        t = np.arange(t0, tn, 1/fs)
        y = [g(i-tau) for i in t]
    elif g == a1b.gabore or g == a1b.gaboro:
        t = np.arange(-T/2, T/2, 1/fs)
        y = g(t, **kwargs)
        if len(t) > 0:
            tb = np.arange(t0, -T/2+tau, 1/fs)
            ta = np.arange(T/2+tau, tn, 1/fs)
            t = np.concatenate((tb, t+tau, ta))
            y = np.concatenate((np.zeros(len(tb)), y, np.zeros(len(ta))))
    else:
        t = np.arange(t0, T, 1/fs)
        y = g(t, **kwargs)
        if len(t) > 0:
            tb = np.arange(t0+1/fs, tau, 1/fs)
            ta = np.arange(t[-1]+tau+1/fs, tn, 1/fs)
            t = np.concatenate((tb, t+tau, ta))
            y = np.concatenate((np.zeros(len(tb)), y, np.zeros(len(ta))))
    return t, y

def plot_stem(t, y, title:str="set me", tunits:str="sec"):
    for i in range(len(y)):
        plt.plot([t[i], t[i]], [0, y[i]], '#1f77b4')
        plt.scatter(t[i], y[i], c='#1f77b4', s=15)
    plt.xlabel(f"time $t$, ({tunits})", fontsize=16)
    plt.ylabel("amplitude", fontsize=16)
    plt.title(title, fontsize=18)
    plt.show()

# 3. Noise and SNR
## 3a. energy, power, and snr
def energy(x):
    return np.linalg.norm(x)**2

def power(x):
    return energy(x)/len(x)

def snr(Ps, Pn):
    return Ps/Pn

# 3b. Noisy signals
def noisysignal(t0, tn, g, fs:int=1, tau:float=0.0, T:float=0.0, s:float=0.0, tscale:float=1, ntype:str="gaussian", **kwargs):
    t, y = gensignal(t0=t0, tn=tn, g=g, fs=fs, tau=tau, T=T, tscale=tscale, **kwargs)
    n = np.random.normal(loc=0, scale=s, size=len(y))
    if ntype.lower() == "uniform":
        n = np.random.uniform(-1, 1, size=len(y))
    return t, y, n

def plot_noisysignal(t, y, title:str="set me", tunits:str="sec", plot_type:str="line"):
    if plot_type.lower() == "line":
        plt.plot(t, y, linewidth=0.5)
    elif plot_type.lower() == "stem":
        for i in range(len(y)):
            plt.plot([t[i], t[i]], [0, y[i]], '#1f77b4')
            plt.scatter(t[i], y[i], c='#1f77b4', s=5)
    else:
        print("plot type not supported")
        return
    plt.xlabel(f"time $t$, ({tunits})", fontsize=16)
    plt.ylabel("amplitude", fontsize=16)
    plt.title(title, fontsize=18)
    plt.show()

# 3c. Noise level specified by SNR
def snr2sigma(x, xrange:int=None, snr:int=10):
    if xrange:
        x = x[:xrange]
    Px = power(x)
    return math.sqrt(Px/pow(10, snr/10))

# 3d. Estimating SNR
def extent(y, th:float=0.01):
    first = last = -1
    for i in range(len(y)):
        if abs(y[i]) > th and first == -1:
            first = last = i
        elif abs(y[i]) > th:
            last = i
    return first, last
