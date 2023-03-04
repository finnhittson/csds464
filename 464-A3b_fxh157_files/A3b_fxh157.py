import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../464-A3a_fxh157_files/')
import A3a_fxh157 as a3a

# 1. Filtering
## 1b. Implementation
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

def randprocess(N, s:float=1.0):
    x = []
    for i in range(N):
        if i-1 >= 0:
            x.append(np.random.normal(loc=x[i-1], scale=s))
        else: x.append(np.random.normal(loc=0, scale=s))
    return x

def plot_movingavg(rand, avg=None, filtered=None, t=None, shift:float=None, title:str="Random Process w/ Moving Average", tunits:str="sec"):
    if t is None:
        t = list(range(len(rand)))
    if avg is not None:
        plt.plot(t, avg, 'r', linewidth=2, label="Moving Average")
    if filtered is not None:
        plt.plot(t, filtered, '#4DB399', linewidth=2, label="Filtered")
    if shift is not None:
        t = t + shift*np.ones(len(t))
    plt.plot(t, rand, '#1f77b4', linewidth=0.5, label="Random Process")
    plt.legend()
    plt.xlabel(f"time, (sec)")
    plt.ylabel("relative value")
    plt.title(title)
    plt.show()

# 2. IIR Filters
## 2a. Implementation
def filterIIR(x, a:list=None, b:list=None):
    y = []
    for n in range(len(x)):
        y.append(0)
        for k in range(len(b)):
            if n-k >= 0:
                y[-1] += b[k] * x[n-k]
        for k in range(len(a)):
            if n-k-1 >= 0:
                y[-1] -= a[k] * y[n-k-1]
    return y

def plot_filter_grid(
	g, a:list=None, b:list=None,
	rows:int=4, cols:int=4, fs:int=2000, 
	tau:float=0, T:float=0.1, tscale:float=1.0, 
	s:list=None, f:list=None
	):
    if cols != len(s) or rows != len(f):
        print("nope!")
        return
    #t=0, g=a1b.sinewave, fs=2000, tau=0, T=0.1, s=0.1, tscale=1, f=100
    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            t, x, n = a3a.noisysignal(t=0, g=g, fs=fs, tau=tau, T=T, s=s[col], tscale=tscale, f=f[row])
            filtered = filterIIR(x=x+n, a=a, b=b)
            axs[row,col].plot(t, x+n)
            axs[row,col].plot(t, filtered)
            #axs[row,col].set_title()
            #axs[row,col].set_xticks(np.arange(-0.08, 0.081, 0.04))

    # figure formatting
    fig.supxlabel("Time $t$, (sec)")
    fig.supylabel("relative value")
    #fig.suptitle("Gabor Functions", fontsize=18)
    #plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()