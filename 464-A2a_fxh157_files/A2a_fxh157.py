import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import random
import math

#
def genwaveform(N:int=100, a:float=0.1, A:float=1.0, s:float=1.0, noisetype:str="Gaussian"):
    waveform = []
    si = []
    for i in range(N):
        rand_val = np.random.normal(loc=0, scale=s)
        if noisetype.lower() == "uniform":
            rand_val = random.uniform(-s/2, s/2)
            if random.uniform(0,1) < a:
                rand_val += A
                si.append(i)
        elif random.uniform(0,1) < a:
            rand_val += A
            si.append(i)
        waveform.append(rand_val)
    return waveform, si

def plot_gauss_noise(waveform, si, noisetype:str="Gaussian", th:float=None):
    plt.plot(list(range(len(waveform))), waveform)
    loc_vals = []
    if th is not None:
        plt.plot([0,len(waveform)], [th, th], '--k')
        tpx = []
        tpy = []
        tnx = []
        tny = []
        fpx = []
        fpy = []
        fnx = []
        fny = []
        si_idx = 0
        for idx, val in enumerate(waveform):
            if val >= th and si_idx < len(si) and si[si_idx] == idx:
                tpx.append(idx)
                tpy.append(waveform[idx])
                si_idx += 1
            elif val >= th and si_idx < len(si) and si[si_idx] != idx:
                fpx.append(idx)
                fpy.append(waveform[idx])
            elif val <= th and si_idx < len(si) and si[si_idx] == idx:
                fnx.append(idx)
                fny.append(waveform[idx])
                si_idx += 1
            elif val <= th and si_idx < len(si) and si[si_idx] != idx:
                tnx.append(idx)
                tny.append(waveform[idx])
        plt.scatter(tpx, tpy, c='r', marker="o", zorder=10, data="tp", edgecolor='k', label="tp")
        plt.scatter(fpx, fpy, c='y', marker="v", zorder=10, data="fp", edgecolor='k', label="fp")
        plt.scatter(fnx, fny, c='g', marker="s", zorder=10, data="fn", edgecolor='k', label="fn")
        plt.legend(loc="lower right")
    else:
        for loc in si:
            loc_vals.append(waveform[loc])
        plt.scatter(si, loc_vals, c='r', s=10, zorder=10)
    plt.xlabel("Time $t$, (sec)", fontsize=16)
    plt.ylabel("Amplitude, (relative)", fontsize=16)
    plt.title(f"Randomly Occuring Events in {noisetype.lower().capitalize()} Noise", fontsize=18)
    plt.show()

def detectioncounts(si, y, theta):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    si_idx = 0
    for idx, val in enumerate(y):
        if val >= theta and si_idx < len(si) and si[si_idx] == idx:
            tp += 1
            si_idx += 1
        elif val >= theta and si_idx < len(si) and si[si_idx] != idx:
            fp += 1
        elif val <= theta and si_idx < len(si) and si[si_idx] == idx:
            fn += 1
            si_idx += 1
        elif val <= theta and si_idx < len(si) and si[si_idx] != idx:
            tn += 1
    return (tp, fn, fp, tn)

def falsepos(th:float=0.5, s:float=1.0, noisetype:str="Gaussian"):
    if noisetype.lower() == "uniform":
        return 1 - th
    return 1 - norm.cdf(th, loc=0, scale=s)

def falseneg(th:float=0.5, A:float=1.0, s:float=1.0, noisetype:str="Gaussian"):
    if noisetype.lower() == "uniform":
        return 1 - th
    return norm.cdf(th-A, loc=0, scale=s)

th=1
s=1.0
A=0.3
noisetype="Gaussian"
falsepos = falsepos(th=th, s=s, noisetype=noisetype)
falseneg = falseneg(th=th, A=A, s=s, noisetype=noisetype)
print(f"predicted false positive rate: {round(falsepos,3)}")
#print(f"false negative rate: {round(falseneg,3)}")

waveform, si = genwaveform(A=0.3)
tp, fn, fp, tn = detectioncounts(si, waveform, theta=th)
print(f"actual false positive rate: {round(fp/(fp+tn),3)}")