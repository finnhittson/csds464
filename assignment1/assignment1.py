import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from itertools import chain

def sinewave(t, f: float=1.0, d: float=0.0) -> float:
	if isinstance(t, int) or isinstance(t, float):
		return math.sin(2 * math.pi * f * (t + d))
	else:
		return [math.sin(2 * math.pi * f * (i + d)) for i in t]


def gabore(t, f: float=1.0, a: float=1.0, d: float=0.0) -> float:
	return a * math.exp(-t**2 / (2 * sigma**2)) * math.cos(2 * math.pi * f * (t + d))


def gaboro(t, f: float=1.0, a: float=1.0, d: float=math.pi/2) -> float:
	return a * math.exp(-t**2 / (2 * sigma**2)) * math.cos(2 * math.pi * f * (t + d))

'''
def gabor_norm(sample_f, f: float=1.0, a: float=1.0, sigma, d: float=0.0):


def gabore_norm(sample_f, f: float=1.0, a: float=1.0, sigma, d: float=0.0):


def gaboro_norm(sample_f, f: float=1.0, a: float=1.0, sigma, d: float=math.pi/2):
	times = list(range(0, 1, sample_f))
'''

def gammatone(t, f: float=1.0, d: float=0.0, n: float=4.0) -> float:
	gt = np.array()
	b = 25.1693*(4.37*f/1000 + 1)
	for i in range(t):
		gt.append(pow(t, n-1)*math.exp(-2*math.pi*b*i)*math.cos(2*math.pi*f*(t+d)))
	return gt / math.sqrt(np.linalg.norm(gt))


def localmaxima(signal):
	maxima_indices = []
	for i in range(1, len(signal)-1):
		if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
			maxima_indices.append(i)
	return maxima_indices


def crossing(signal, threshold, negpos: bool=True, posneg: bool=False):
	threshold_cross_idx = []
	for idx, val in enumerate(signal):
		if negpos and posneg and val != threshold:
			threshold_cross_idx.append(idx)
		if negpos and not posneg and val > threshold:
			threshold_cross_idx.append(idx)
		if not negpos and posneg and val < threshold:
			threshold_cross_idx.append(idx)
	return threshold_cross_idx


def envelope(y, nblocks: int=10):
	blockindices = list(range(0, len(y), nblocks))
	ylower = [min(y[b_idx: b_idx+nblocks]) for b_idx in blockindices]
	yupper = [max(y[b_idx: b_idx+nblocks]) for b_idx in blockindices]
	return ylower, yupper, blockindices


def read_wav_file(filepath):
	_, data = wavfile.read(filepath)
	return data


def plot_sinwave(start: int=-5, stop: int=5, step: int=0.01, f: float=1.0, d: float=0.0):
	t = np.arange(start, stop + step, step)
	y = [sinewave(t=time, f=f, d=d) for time in t]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')

	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	plt.xlim(start, stop)
	plt.ylim(-2, 2)

	plt.plot(t, y)
	plt.show()


def plot_envelope(y, nblocks):
	ylower, yupper, _ = envelope(y, nblocks)
	t = np.arange(0, 5, 5/(len(ylower) + len(yupper)))
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	y = list(chain.from_iterable(zip(ylower, yupper)))
	plt.plot(t, y)
	plt.show()


filepath = "C:/Users/hitts/Desktop/speech.wav"
data = read_wav_file(filepath)
plot_envelope(data, 1000)