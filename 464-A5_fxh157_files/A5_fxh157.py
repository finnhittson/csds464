import matplotlib.pyplot as plt
import numpy as np

# (1)
def plot_circle(k, N):
    ns = np.array(list(range(0, N+1)))
    x = [np.cos(2*np.pi*k/N*n) for n in ns]
    y = [np.sin(2*np.pi*k/N*n) for n in ns]

    plt.figure().set_figheight(5)
    plt.figure().set_figwidth(5)
    plt.plot(x, y)
    plt.scatter(x, y, c='r', s=5, zorder=10)
    plt.title(f"unit circle, $N={N}$, $k={k}$")
    plt.show()

def w(n, k, N):
    return np.exp(2j*np.pi*k*n/N)

def plotw(k, N, show_cont:bool=False):
    ns = list(range(0, N+1))
    vals = np.array([w(n, k, N) for n in ns])
    x = vals.real
    y = vals.imag

    plt.figure().set_figheight(5)
    plt.figure().set_figwidth(5)
    fig = plt.stem(x, y, basefmt="")

    if show_cont:
        val = np.array([w(n=n, k=1, N=N) for n in range(0, N+1)])
        plt.plot(val.real, val.imag, zorder=0)

    plt.show()

# (2)
def fourier_matrix(N):
    return np.matrix([[w(n=n, k=k, N=N) for n in range(N)] for k in range(N)])

# (3)
def eatdogshit(y, k, l, M, N):
    out = 0
    for m in range(M):
        for n in range(N):
            out += y[n, m]*np.exp(-2j*np.pi*(k*m/M + l*n/N))
    return out / np.sqrt(M*N)

def fourier2d(y):
    N = y.shape[0]
    M = y.shape[1]
    return np.array([[eatdogshit(y=y, k=k, l=l, M=M, N=N) for k in range(M)] for l in range(N)])

def plot_2dfourier(fft1, fft2, title1:str="set me!", title2:str="set me!"):
    plt.subplot(121)
    plt.matshow(fft1, fignum=False)
    plt.title(title1)

    plt.subplot(122)
    plt.matshow(fft2, fignum=False)
    plt.title(title2)

    plt.show()