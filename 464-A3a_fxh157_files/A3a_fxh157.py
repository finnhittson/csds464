import math
import numpy as np
import matplotlib.pyplot as plt

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
    #'''
    for st in sample_times:
        y = g(t=st, **kwargs)/norm
        plt.plot([st, st], [0, y], 'r')
        plt.scatter(st, y, c='r', s=15, zorder=10)
    #'''
    plt.xlim([t[0], t[-1]])
    plt.xlabel(f"time $t$, ({tunits})", fontsize=16)
    plt.ylabel("amplitude", fontsize=16)
    plt.title(title, fontsize=18)
    plt.show()

# 2. Signals
## 2a. Delta and step functions
def d(t, fs:int=1):
    return np.array([1 if round(i, 3) == 0 else 0 for i in t])
    
def u(t):
    return np.array([1 if i >= 0 else 0 for i in t])

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
def gensignal(t, g, fs:int=1, tau:float=1.0, T:float=1.0, tscale:float=0.001, **kwargs):
    if g == d:
        t = np.arange(t, t+T+tau, 1/(tscale*fs))
        y = g(t, **kwargs)
        return t + np.ones(len(t))*tau, y
    t = np.arange(t+tau, T+tau, 1/(tscale*fs))
    y = g(t, **kwargs)
    return t, y

def plot_stem(t, y, title, time_units):
    for i in range(len(y)):
        plt.plot([t[i], t[i]], [0, y[i]], '#1f77b4')
        plt.scatter(t[i], y[i], c='#1f77b4', s=15)
    plt.xlabel(f"time $t$, ({time_units})", fontsize=16)
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
def noisysignal(t, g, fs, tau, T, s, tscale:float=0.001, **kwargs):
    t, signal = gensignal(t=t, g=g, fs=fs, tau=tau, T=T, tscale=tscale, **kwargs)
    noise = np.random.normal(loc=0, scale=s, size=len(signal))
    return t, signal, noise

def plot_noisysignal(t, y, title, tunits):
    plt.plot(t, y, linewidth=0.5)
    plt.xlabel(f"time $t$, ({tunits})", fontsize=16)
    plt.ylabel("amplitude", fontsize=16)
    plt.title(title, fontsize=16)
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

def extend(t, y, n, fs, T, size, s):
    noise_before = np.random.normal(loc=0, scale=s, size=size)
    noise_after = np.random.normal(loc=0, scale=s, size=size)
    noise = np.concatenate((noise_before, n, noise_after))

    zeros = np.zeros(size)
    signal = np.concatenate((zeros, y, zeros))
    time = np.linspace(0, T + 2*T*size/len(y), len(signal))

    return time, signal, noise