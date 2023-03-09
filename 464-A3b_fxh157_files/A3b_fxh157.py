import numpy as np
import random
import matplotlib.pyplot as plt

import sys
sys.path.append('../464-A3a_fxh157_files/')
import A3a_fxh157 as a3a

# 1. Filtering
def movingavg(x, l:float=0.5):
    if l > 1:
        print("lambda needs to be between 0 and 1.")
        return None
    y = []
    for n in range(len(x)):
        running_sum = (1-l) * x[n]
        if n-1 >= 0:
            running_sum += l*y[n-1]
        y.append(running_sum)
    return y

def randprocess(N, s:float=1.0, ntype:str="gaussian"):
    x = []
    if ntype.lower() == "gaussian":
        for i in range(N):
            if i-1 >= 0:
                x.append(np.random.normal(loc=x[i-1], scale=s))
            else: x.append(np.random.normal(loc=0, scale=s))
    if ntype.lower() == "uniform":
        for i in range(N):
            if i-1 >= 0:
                x.append(random.uniform(-1, 1))
            else: x.append(random.uniform(-1, 1))
    return x

def plot_movingavg(rand, avg=None, filtered=None, t=None, shift:float=None, title:str="Random Process w/ Moving Average", tunits:str="sec", label:str="Filtered"):
    if t is None:
        t = list(range(len(rand)))
    plt.plot(t, rand, '#1f77b4', linewidth=0.5, label="Random process")
    if shift is not None:
        t = t + shift*np.ones(len(t))
    if avg is not None:
        plt.plot(t, avg, 'r', linewidth=2, label="Moving avg")
    if filtered is not None:
        plt.plot(t, filtered, 'r' if avg is None else '#4DB399', linewidth=2, label=label)
    plt.legend()
    plt.xlabel(f"time, ({tunits})", fontsize=16)
    plt.ylabel("relative value", fontsize=16)
    plt.title(title, fontsize=18)
    plt.show()

# 2. IIR Filters
def filterIIR(x, a:list=None, b:list=None):
    y = []
    for n in range(len(x)):
        y.append(0)
        for k in range(len(b)):
            if n-k >= 0:
                y[-1] += b[k] * x[n-k]
        for k in range(len(a)):
            if n-k >= 1:
                y[-1] -= a[k] * y[n-k-1]
    return y

def plot_filter_grid(
	g, a:list=None, b:list=None,
	rows:int=4, cols:int=4, t0:float=0.0, tn:float=0.1,
    fs:int=2000, tau:float=0, T:float=0.1, 
    tscale:float=1.0, s:list=None, f:list=None
	):
    if cols != len(s) or rows != len(f):
        print("nope!")
        return
    fig, axs = plt.subplots(rows, cols, figsize=(7, 7))
    fig.tight_layout(pad=0.1)
    for row in range(rows):
        for col in range(cols):
            t, x, n = a3a.noisysignal(t0=t0, tn=tn, g=g, fs=fs, tau=tau, T=T, s=s[row], tscale=tscale, f=f[col])
            filtered = filterIIR(x=x+n, a=a, b=b)
            axs[row,col].plot(t, x+n)
            axs[row,col].plot(t, filtered)
            axs[row,col].set_ylim([-2,2])
            axs[row,col].set_yticks([-2,0,2])
            axs[row,col].set_xticks([0,0.05,0.1])
            if col == 0:
                axs[row,col].set_ylabel(f"$\sigma={s[row]}$")
            if row == 3:
                axs[row,col].set_xlabel(f"$f={f[col]}Hz$")
    plt.show()

def freqpower(g, a, b, t0:float=0.0, tn:float=1.0, fs:int=2000, tau:float=0.0, T:float=0.1, s:float=0.1, tscale:float=1.0):
    p = []
    freqs = np.arange(0, fs/2, 1)
    for f in freqs:
        t, x, n = a3a.noisysignal(t0=t0, tn=tn, g=g, fs=fs, tau=0, T=0.1, s=0.1, tscale=1, f=f)
        filtered = filterIIR(x=x+n, a=a, b=b)
        p.append(a3a.power(filtered))
    return freqs, p

# 3. The impulse response function
def impulse(x, step:int=1):
    y = []
    t = []
    idx = 0
    while idx < len(x):
        t.append(idx)
        y.append(0)
        for k in range(len(x)):
            y[-1] += x[k]*a3a.d(t[-1]-k)
        idx += 1*step
    return np.array(t), y

def plot_impulse(t, y:list=None, t0:list=None, x0:list=None, rand:list=None, shift:float=None, title:str="Impulse function", tunits:str="sec", label:str="filtered"):
    if rand is not None:
        plt.plot(t0, rand, label="noise", linewidth=1)
    if shift is not None:
        t = t + shift*np.ones(len(t))
        t0 = t0 + shift*np.ones(len(t0))
    if x0 is not None and t0 is not None:
        plt.plot(t0, x0, '#4DB399', linewidth=5, label=label, zorder=10)
    if y is not None:
        for i in range(len(y)):
            plt.plot([t[i], t[i]], [0, y[i]], 'r', label='impulse' if i == 0 else "", zorder=10)
            plt.scatter(t[i], y[i], c='r', s=5, zorder=10)
    plt.xlabel(f"time, ({tunits})", fontsize=16)
    plt.ylabel("relative value", fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.show()

# 4. Filtering with convolution
def convolve(x, h, h0:int=0):
    y = []
    for n in range(0, len(x)):
        y.append(0)
        for k in range(len(h)):
            k = n+h0-k
            if len(x) > k >= 0:
                y[-1] += x[k]*h[n-k+h0]
    return y

def plot_convolution(t, y:list=None, t0:list=None, x0:list=None, rand:list=None, shift:float=None, title:str="Impulse function", tunits="sec", label1:str="filtered", label2="convolution"):
    if rand is not None:
        plt.plot(t0, rand, label="noise", linewidth=1)
    if shift is not None:
        t = t + shift*np.ones(len(t))
        t0 = t0 + shift*np.ones(len(t0))
    if x0 is not None and t0 is not None:
        plt.plot(t0, x0, '#4DB399', linewidth=5, label=label1, zorder=10)
    if y is not None:
        plt.plot(t, y, 'r', label=label2, zorder=10)
    plt.xlabel(f"time, ({tunits})", fontsize=16)
    plt.ylabel("relative value", fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend().set_zorder(10)
    plt.show()