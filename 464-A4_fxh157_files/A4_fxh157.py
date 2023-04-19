import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import numpy as np
import math

import sys
sys.path.append('../464-A1b_fxh157_files/')
import A1b_fxh157 as a1b

import sys
sys.path.append('../464-A3b_fxh157_files/')
import A3b_fxh157 as a3b

def read_wav(filepath):
    sr, data = wavfile.read(filepath)
    return sr, data

# (1) Spectral Structure
def harmonic(t, f:int=1, alist:list=[1], phase_list:list=[0]):
    val = 0
    for idx in range(len(alist)):
        val += alist[idx]*math.cos(2*math.pi*f*(idx+1)*t+phase_list[idx])
    return val

def cosine(t, f:list=[1], alist:list=[1], phase_list:list=[0]):
    val = 0
    for idx in range(len(alist)):
        val += alist[idx]*math.cos(2*math.pi*f[idx]*t+phase_list[idx])
    return val

def plot_harmonics(t, g, f, n=None, alist:list=[1], phase_list:list=[0], title:str="set me"):
    y = np.array([g(t=i, f=f, alist=alist, phase_list=phase_list) for i in t])
    if n is not None and len(y) == len(n):
        y += n
    fig, axs = plt.subplots(1,2, figsize=(10,2))
    fig.subplots_adjust(wspace=0.3)
    if isinstance(f, int) or isinstance(f, float):
        axs[0].stem([f*(i+1) for i in range(len(alist))], basefmt="")
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

# (2) Noise
def bandpass_noise(t, f:int=100, sigma_k:float=0.01, sigma_n:float=0.1, ntype:str="gaussian", plot_kernel:bool=True):
    kernel = a1b.gabore(t=t, f=f, a=1, d=0, sigma=sigma_k)
    noise = np.random.normal(loc=0, scale=sigma_n, size=1000)
    if ntype.lower() == "uniform":
        noise = np.random.uniform(-1, 1, size=1000)
    conv = np.convolve(noise, kernel)
    
    if plot_kernel:
        plt.figure().set_figheight(2)
        plt.plot(t, kernel)
        plt.title(f"gabor kernel, $f={f}$, $\sigma={sigma_k}$")

    plt.figure().set_figheight(2)
    plt.plot(noise)
    plt.title(f"{ntype} noise, $\sigma={sigma_n}$")
    if ntype.lower() == "uniform":
        plt.title(f"{ntype} noise")
    
    plt.figure().set_figheight(2)
    plt.plot(conv)
    plt.title("bandpass noise")

    plt.show()

# (4) Implementation and Application
def convolve(x, y):
    out1 = np.zeros(len(x))
    for n in range(len(x)):
        for k in range(len(y)):
            if len(y) > n-k >= 0:
                out1[n] += y[n-k]*x[k]

    out2 = np.zeros(len(x)-1)
    for n in range(len(x)-1):
        for k in range(len(y)):
            if len(y) > n+len(y)-k >= 0:
                out2[n] += y[n+len(y)-k]*x[k]

    return np.concatenate((out1, out2))

def convolve1(x, h):
    y = np.zeros(len(x)*2-1)
    for n in range(-len(x)+1, len(x)):
        for k in range(len(h)):
            if len(x) > n-k >= 0:
                y[n+len(x)-1] += x[n-k]*h[k]
    return y

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
    pxy = np.zeros(len(x)*2-1)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    for n in range(-len(x)+1, len(x)):
        for k in range(len(y)):
            if len(x)> k-n >= 0:
                pxy[n+len(x)-1] += x[k-n]*y[k]
        if normalize:
            pxy[n+len(x)-1] /= norm_x * norm_y
    return pxy

def pitch_estimate(data, N:int=1000, plot_data:bool=True, plot_autoc:bool=True):
    data_s = data[:N]
    x = [i/44100 for i in range(-len(data_s)+1, len(data_s))]

    auto_c = autocorr(x=data_s, normalize=True)

    peaks, _ = scipy.signal.find_peaks(auto_c)
    sorted_auto_c = np.argsort(auto_c[peaks])
    twopeak = abs(x[peaks[sorted_auto_c[-2]]] - x[peaks[sorted_auto_c[-1]]])
    ff = 1/twopeak
    print(f"pitch estimate: {ff} Hz")
    if plot_data:
        plt.figure().set_figheight(2)
        plt.plot(data_s)
        plt.title("original signal")

    if plot_autoc:
        plt.figure().set_figheight(2)
        plt.title("autocorrelation")
        plt.plot(x, auto_c)

    plt.show()

    return ff

def estimate_time_delay(x, y):
    cross_corr = crosscorr(x=x, y=y, normalize=True)
    lag = np.argmax(cross_corr) - len(x) + 1
    print(f"time delay: {lag} lag samples")

    plt.figure().set_figheight(2)
    plt.plot(cross_corr, label="cross-correlation")
    plt.scatter(np.argmax(cross_corr), max(cross_corr), c='r', label="max correlation", zorder=10)
    plt.legend()
    plt.show()

    return lag

# (5) Spectral Analysis
def spectral_analysis(data, f:int=None, alist:list=None, scale:int=1, N:int=None, x:int=-2, plot_data:bool=True):
    if N is not None:
        data = data[:N]

    fft_data = scipy.fft.rfft(data)
    freqs = scipy.fft.rfftfreq(n=data.size, d=1/len(data)) * scale

    a = abs(fft_data)/len(fft_data)

    idxs = a > pow(10, x)
    freqs = freqs[idxs]
    a = a[idxs]

    if plot_data:
        plt.figure().set_figheight(2)
        plt.plot(data)

    if f and alist:
        orig_freqs = [f*(i+1) for i in range(len(alist))]
        plt.figure().set_figheight(2)
        plt.stem(orig_freqs, alist, basefmt="")

    plt.figure().set_figheight(2)
    plt.stem(freqs, a, basefmt="")
    plt.show()