import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def load_ecg(path):
    data = pd.read_csv(path, header=None, sep=None, engine="python")
    ecg = pd.to_numeric(data.iloc[:,0], errors="coerce")
    ecg = ecg.interpolate().fillna(method="bfill").fillna(method="ffill").values
    return ecg

def bandpass_filter(signal, fs=1000, low=0.5, high=40, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(filtered, fs=1000):
    energy = filtered**2
    smooth = np.convolve(energy, np.ones(int(fs*0.05))/int(fs*0.05), mode='same')
    peaks, _ = find_peaks(smooth, distance=int(0.2*fs), height=np.mean(smooth)*1.5)
    return peaks

def extract_beats(signal, peaks, fs=1000, pre=0.2, post=0.4):
    n_pre, n_post = int(pre*fs), int(post*fs)
    beats = []
    for p in peaks:
        if p > n_pre and p + n_post < len(signal):
            beats.append(signal[p-n_pre:p+n_post])
    return np.array(beats)
