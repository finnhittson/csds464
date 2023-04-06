import matplotlib.pyplot as plt
import numpy as np
import math

def ping():
    print("pong")

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

def plot_harmonics(t, g, f, alist:list=[1], phase_list:list=[0], title:str="set me"):
    y = [g(t=i, f=f, alist=alist, phase_list=phase_list) for i in t]
    fig, axs = plt.subplots(1,2, figsize=(7,2))
    if isinstance(f, int) or isinstance(f, float):
        axs[0].stem([f*(i+1) for i in range(len(alist))], basefmt=" ")
    else:
        axs[0].stem(f, basefmt=" ")
    axs[0].set_title("frequencies")
    axs[0].set_xticks(ticks=list(range(0, len(alist))), labels=list(range(1, len(alist)+1)))
    axs[1].plot(t, y)
    axs[1].set_title(title)
    plt.show()