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
    plt.figure().set_figheight(2)
    plt.subplot(121)
    if isinstance(f, int) or isinstance(f, float):
        plt.stem([f*(i+1) for i in range(len(alist))], basefmt=" ")
    else:
        plt.stem(f, basefmt=" ")
    plt.subplot(122)
    plt.plot(t, y)
    plt.title(title)
    plt.show()