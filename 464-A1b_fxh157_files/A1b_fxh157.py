import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from itertools import chain


# returns computed value along sinewave or list of values
#   t: int or list of times to compute sinewave at
#   f: sinewave frequency
#   d: sinewave delay in seconds
#   a: sinewave amplitude
def sinewave(t, f: float=1.0, d: float=0.0, a: float=1.0):
	if isinstance(t, int) or isinstance(t, float): # returns single computed value
		return a*math.sin(2*math.pi*f*(t + d))
	else: # returns list of computed values
		return [a*math.sin(2*math.pi*f*(i + d)) for i in t]


# plots sinewave function
#   start: left most x-axis integer to compute
#   stop: right most x-axis integer to compute
#   step: increment from start and stop values
#   tick_step: x-axis increment size
#   ...
#   show_delay: ternary of whether to plot sinewave without delay
#   show_max: ternary of whther to show mark max of sinewave
#   threshold: value of threshold to plot
#   negpos: cross threshold from negative to positive
#   posneg: cross threshold from positive to negative
def plot_sinewave(
	start: float=-5, 
	stop: float=5, 
	step: int=0.01, 
	tick_step: float=1.0, 
	f: float=1.0, 
	d: float=0.0, 
	a: float=1.0, 
	show_delay: bool=False, 
	show_max: bool=False,
	threshold: float=None,
	negpos: bool=True,
	posneg: bool=False
	):
    # computes and plots sinewave from input parameters
    t = np.arange(start, stop + step, step)
    y = [sinewave(t=time, f=f, d=d, a=a) for time in t]

    if threshold is not None: # plots threshold
    	plt.plot(t, y, label=f"Sinwave", linewidth='2')
    	plt.plot([start, stop], [threshold, threshold], '--k', label="Threshold")
    	threshold_cross_idx = crossing(y, threshold, negpos=negpos, posneg=posneg)
    	t_thresh = [t[i] for i in threshold_cross_idx]
    	y_thresh = [y[i] for i in threshold_cross_idx]
    	plt.scatter(t_thresh, y_thresh, c='r', s=10, zorder=10, label="Cross Threshold")
    	plt.legend(loc='upper right')
    elif show_delay: # makes delayed sinewave a dashed line
    	plt.plot(t, y, '--', label=f"Delayed function: $d$={round(d, 3)} sec", linewidth='2')
    else: # normal delayed sinewave
    	plt.plot(t, y, label=f"Delayed function: $d$={round(d, 3)} sec", linewidth='2')

    # plots non-shifted sinewave to help visualize phase shift
    if show_delay:
        y_delay = [sinewave(t=time, f=f, d=0.0, a=a) for time in t]
        plt.plot(t, y_delay, label="Non-delayed function", linewidth='2')
        plt.legend()

    # marks sinewave maximum points
    if show_max:
        max_idx = localmaxima(y)
        t_max = [t[i] for i in max_idx]
        y_max = [y[i] for i in max_idx]
        plt.scatter(t_max, y_max, c='r', s=30, label=f'maximum: $a$={a}', zorder=10)
        plt.legend()

    # formatting graph
    plt.xlabel("Time $t$, (sec)", fontsize=16)
    plt.ylabel("$sin(2\\pi ft+\\phi)$", fontsize=16)
    plt.title(f"Sinewave, $f$={f}, $\\phi$={round(2*math.pi*f*d, 3)}, $d$={d} sec", fontsize=18)
    plt.xlim(start, stop)
    plt.xticks(np.arange(start, stop+tick_step/10, step=tick_step))
    plt.ylim(-2*a, 2*a)
    plt.show()


# even gabor function
#   t: time value to compute the gabor value
#   f: gabor/cos frequency
#   a: gabor amplidute
#   d: gabor/cos delay
#   sigma: Gaussian width
def gabore(t, f: float=1.0, a: float=1.0, d: float=0.0, sigma: float=1.0) -> float:
	return a*math.exp(-t**2/(2*sigma**2)) * math.cos(2*math.pi*f*(t + d))


# odd gabor function
# 	...
def gaboro(t, f: float=1.0, a: float=1.0, d: float=math.pi/2, sigma: float=1.0) -> float:
	return a*math.exp(-t**2/(2*sigma**2)) * math.cos(2*math.pi*f*t + d)


# computes the normalization constant of the even gabor function over a range of time
#   start: left most time point to begin measurments at
#   stop: right most time point to end measurments at 
#   sample_f: sampling frequency with which to measure time points at
#	...
def gabore_norm(y: list=None, fs: float=10000.0, f: float=1.0, d: float=0.0, sigma: float=1.0):
	if y is None:
		t = np.arange(-4*sigma, 4*sigma, 1/fs)
		y = np.array([gabore(t=i, f=f, d=d, sigma=sigma) for i in t])
	return np.linalg.norm(y)


# computes the normalization constant of the odd gabor function over a range of time
#	...
def gaboro_norm(y: list=None, fs: float=0.1, f: float=1.0, d: float=math.pi/2, sigma: float=1.0):
	if y is None:
		t = np.arange(-4*sigma, 4*sigma, 1/fs)
		y = np.array([gaboro(t=i, f=f, d=d, sigma=sigma) for i in t])
	return np.linalg.norm(y)


# plots gabor function
#   start: left most x-axis integer to compute
#   stop: right most x-axis integer to compute
#   step: increment from start and stop values
# 	...
def plot_gabor(start: float=-5, stop: float=5, fs: float=10000, f: float=1.0, d: float=0.0, sigma: float=1.0):
	# computes and plots gabor function from input parameters
	t = np.arange(start, stop+1/fs, 1/fs)
	y = np.array([gabore(t=time, f=f, d=d, sigma=sigma) for time in t]) # un-normalized
	a = 1/gabore_norm(y) # normalize
	plt.plot(t, y*a)
	
	# formatting graph
	plt.xlabel("Time $t$, (sec)", fontsize=16)
	plt.ylabel("$a\\cdot exp\\left(\\frac{-t^2}{2\\sigma^2}\\right)cos(2\\pi ft+\\phi)$", fontsize=16)
	plt.title(f"Gabor Function, $a$={round(a, 3)}, $f$={f}, $\\phi$={round(d, 3)}, $\\sigma$={sigma}", fontsize=18)
	plt.show()


# plots grid of gabor functions for lists of input values
def plot_gabor_grid(rows: int=2, cols: int=2, start: float=-5, stop: float=5, fs: int=10000, f: list=[], d: list=[], sigma: list=[]):
	# only plots if correct number of inputs are inputted
	if cols*rows != len(f) or cols*rows != len(d) or cols*rows != len(sigma):
		print("Please enter correct number of values for f, d, and/or, sigma")
		return

	# generating plots
	fig, axs = plt.subplots(rows, cols)
	t = np.arange(start, stop, 1/fs)
	idx = 0
	for row in range(rows):
		for col in range(cols):
			y = np.array([gabore(t=time, f=f[idx], d=d[idx], sigma=sigma[idx]) for time in t])
			a = gabore_norm(fs=fs, f=f[idx], d=d[idx], sigma=sigma[idx])
			axs[row,col].plot(t,y*a)
			axs[row,col].set_title(f"$f$={f[idx]}, $a$={round(a, 3)}, $\\phi$={round(d[idx], 3)}, $\\sigma$={round(sigma[idx],3)}", fontsize=6)
			axs[row,col].set_xticks(np.arange(-0.08, 0.081, 0.04))
			idx += 1
	
	# figure formatting
	fig.supxlabel("Time $t$, (sec)", fontsize=16)
	fig.supylabel("$a\\cdot exp\\left(\\frac{-t^2}{2\\sigma^2}\\right)cos(2\\pi ft+\\phi)$", fontsize=16)
	fig.suptitle("Gabor Functions", fontsize=18)
	plt.subplots_adjust(hspace=0.4, wspace=0.3)
	plt.show()


# plots a gabor function as a stem plot
def plot_gabor_stem(start: float=-5, stop: float=5, fs: float=16, f: float=0.01, d: float=0.0, sigma: float=1.0):
	# computes and plots gabor function from input parameters
	t = np.arange(-4*sigma, 4*sigma, 1/fs)
	y = np.array([gabore(t=time, f=f, d=d, sigma=sigma) for time in t]) # un-normalized
	a = gabore_norm(y) # normalization constant
	
	for idx, val in enumerate(y):
		plt.plot([t[idx], t[idx]], [0,val*a], 'b', linewidth=0.5)
		plt.scatter([t[idx]], [val*a], s=20, c='b')
	
	# formatting graph
	plt.xlabel("Time $t$, (sec)", fontsize=16)
	plt.ylabel("$a\\cdot exp\\left(\\frac{-t^2}{2\\sigma^2}\\right)cos(2\\pi ft+\\phi)$", fontsize=16)
	plt.title(f"Gabor Function, $a$={round(a, 3)}, $f$={f}, $\\phi$={round(d, 3)}, $\\sigma$={sigma}", fontsize=18)
	plt.show()


# returns gammatone point(s)
# 	t: time list or floating point
#		f: gammatone frequency
#		d: delay in seconds
#		n: shape parameter
#		a: amplitude
def gammatone(t, f: float=1.0, d: float=0.0, n: float=4.0, a: float=1.0):
	b = (1.019*24.7*(4.37*f/1000 + 1))
	if isinstance(t, int) or isinstance(t, float): # computes gammatone for list of times
		return a*pow(t, n-1)*math.exp(-2*math.pi*b*t)*math.cos(2*math.pi*f*(t + d))
	else: # computes single value of gammatone
		gt = []
		for time in t:
			gt.append(pow(time, n-1)*math.exp(-2*math.pi*b*time)*math.cos(2*math.pi*f*time + d))
		return np.array(gt) / math.sqrt(np.linalg.norm(gt))


# returns norm of gammatone function
def gammatone_norm(y: list=None, stop: float=100, sf: float=10000, f: float=1.0, d: float=0.0, n: float=4.0):
	if y is None:
		t = np.arange(0, stop, 1/sf)
		y = np.array([gammatone(t=i, f=f, d=d, n=n) for i in t])
	return np.linalg.norm(y)


# plots gammatone function
#   start: left most x-axis integer to compute
#   stop: right most x-axis integer to compute
#   step: increment from start and stop values
# 	tick_step: x-axis increment value
# 	...
def plot_gammatone(start: int=0, stop: int=100, step: int=0.0001, tick_step: float=1.0, f: float=1.0, d: float=0.0, n: float=4.0, a: float=1.0, mark_max: bool=False):
	# plots gammatone function
	t = np.arange(start, stop + step, step)
	y = gammatone(t=t.tolist(), f=f, d=d, n=n)
	a = 1/gammatone_norm(y)
	plt.plot(t, y*a, label='Gammatone')

	if mark_max:
		max_idx = localmaxima(y*a)
		y_max = np.array([y[i] for i in max_idx])
		t_max = [t[i] for i in max_idx]
		plt.scatter(t_max, a*y_max, c='r', s=20, zorder=10, label='Local Maximum')
		plt.legend(loc='upper right')

	# formatting graph
	plt.xlabel("Time $t$, (sec)", fontsize=16)
	plt.ylabel("$at^{n-1}e^{-2\\pi bt}cos(2\\pi ft+\\phi)$", fontsize=16)
	plt.title(f"Gammatone, $f$={f}, $\\phi$={round(d, 3)}, $n$={n}", fontsize=18)
	plt.show()


# returns indicies of local maximas on a 1D signal
#		s: input signal
def localmaxima(s):
	local_max = []
	for i in range(1, len(s)-1):
		if s[i] > s[i-1] and s[i] > s[i+1]:
			local_max.append(i)
	return local_max


# returns indices where signal crosses a threshold
# 	s: input signal
# 	th: threshold value
#		negpos: define threshold direction to be from negative to positive
#		posneg: define threshold direction to be from positive to negative
def crossing(s, th, negpos: bool=True, posneg: bool=False):
	threshold_cross_idx = []
	for idx in range(1, len(s)):
		if negpos and posneg and ((s[idx] >= th and s[idx-1] < th) or (s[idx] < th and s[idx-1] >= th)):
			threshold_cross_idx.append(idx)
		if negpos and not posneg and s[idx] >= th and s[idx-1] < th:
			threshold_cross_idx.append(idx)
		if not negpos and posneg and s[idx] < th and s[idx-1] >= th:
			threshold_cross_idx.append(idx)
	return threshold_cross_idx


# returns maximum and minimum values within the nblocks of the waveform y
#		y: input waveform as a list
#		nblocks: number of regions to search for min and max in
def envelope(y, nblocks: int=10):
	blockindices = list(range(0, len(y), nblocks))
	ylower = [min(y[b_idx: b_idx+nblocks]) for b_idx in blockindices]
	yupper = [max(y[b_idx: b_idx+nblocks]) for b_idx in blockindices]
	return ylower, yupper, blockindices

#10
#print(envelope([5, 5, 2, 3, 4, 3, -6, -9, 0, -3, 9, -7], nblocks=3))

# reads .wav files and returns data
def read_wav_file(filepath):
    sr, data = wavfile.read(filepath)
    return sr, data


# plots envelope
#   y: input waveform or signal
#		nblocks: number of regions to search for min and max in
def plot_envelope(y, nblocks):
    # finds mins and maxes
    ylower, yupper, _ = envelope(y, nblocks)
    t = np.arange(0, 5, 5/len(ylower))
    plt.plot(t, ylower, 'b')
    plt.plot(t, yupper, 'b')

    # graph formatting
    plt.xlabel("Time $t$, (sec)", fontsize=16)
    plt.ylabel("Amplitude, (relative)", fontsize=16)
    plt.title(f"Sound Wave, nblocks={nblocks}", fontsize=18)
    plt.show()


# plots a sound wave
#   s: input signal
#   sr: sample rate (samples/sec)
#   start: starting time
#   stop: stoping time
def plot_sound_wave(s, sr, start, stop):
	i = start * sr
	y = []
	t = []
	while i < stop * sr:
		y.append(s[int(i)])
		t.append(i/sr)
		i += 1
	plt.plot(t, y)
	plt.xlabel("Time $t$, (sec)", fontsize=16)
	plt.ylabel("Amplitude, (relative)", fontsize=16)
	plt.title("Sound Wave", fontsize=18)
	plt.show()


def timetoindex(time, fs):
	return time * fs
