import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import math

def read_wav(filepath):
    sr, data = wavfile.read(filepath)
    return sr, data

# (1) Spectral Structure
def harmonic(t, f:int=1, alist:list=[1], phase_list:list=[0]):
    val = 0
    for idx in range(len(alist)):
        val += alist[idx]*math.cos(f*(idx+1)*t+phase_list[idx])
    return val

def cosine(t, f:list=[1], alist:list=[1], phase_list:list=[0]):
    val = 0
    for idx in range(len(alist)):
        val += alist[idx]*math.cos(f[idx]*t+phase_list[idx])
    return val

def plot_harmonics(t, g, f, n=None, alist:list=[1], phase_list:list=[0], title:str="set me"):
    y = np.array([g(t=i, f=f, alist=alist, phase_list=phase_list) for i in t])
    if n is not None and len(y) == len(n):
        y += n
    fig, axs = plt.subplots(1,2, figsize=(10,2))
    fig.subplots_adjust(wspace=0.3)
    if isinstance(f, int) or isinstance(f, float):
        axs[0].stem([f*(i+1) for i in range(len(alist))], basefmt=" ")
    else:
        axs[0].stem(f, basefmt=" ")
    axs[0].set_title("frequencies")
    axs[0].set_xlabel("count")
    axs[0].set_ylabel("frequency, $Hz$")
    axs[0].set_xticks(ticks=list(range(0, len(alist))), labels=list(range(1, len(alist)+1)))

    axs[1].plot(t, y)
    axs[1].set_title(title)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("amplitude")

    plt.show()

# (4) Implementation and Application
def convolve(x, y):
    out = np.zeros(len(x))
    for n in range(len(x)):
        for k in range(len(y)):
            if len(y) > n+len(y)-k >= 0:
                out[n] += y[n+len(y)-k]*x[k]

    out2 = np.zeros(len(x))
    for n in range(len(x)):
        for k in range(len(y)):
            if  n-k >= 0:
                out2[n] += y[n-k]*x[k]
    return np.concatenate((out2, out))

def autocorr(x, normalize:bool=True):
    pxx = np.zeros(len(x)*2-1)
    norm_sqr = np.linalg.norm(x)**2
    for n in range(-len(x)+1, len(x)):
        for k in range(len(x)):
            if len(x) > k-n >= 0:
                pxx[n+len(x)-1] += x[k-n]*x[k]
        if normalize:
            pxx[n+len(x)-1] /= norm_sqr
    return pxx

def crosscorr(x, y, normalize:bool=True):
    pxy = np.zeros(len(x))
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    for n in range(len(x)):
        for k in range(len(y)):
            if k-n >= 0:
                pxy[n] += x[k-n]*y[k]
        if normalize:
            pxy[n] /= norm_x * norm_y
    return pxy