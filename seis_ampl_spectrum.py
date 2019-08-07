import numpy as np
import matplotlib.pyplot as plt
def ampspec(signal,sr,smooth=False):
    '''
    ampspec (C) aadm 2016
    Calculates amplitude spectrum of a signal with FFT optionally smoothed via cubic interpolation.

    INPUT
    signal: 1D numpy array
    sr: sample rate in ms
    smooth: True or False

    OUTPUT
    freq: frequency
    amp: amplitude
    '''

    SIGNAL = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.size, d=sr*0.001)
    keep = freq>=0
    SIGNAL = np.abs(SIGNAL[keep])
    freq = freq[keep]
    if smooth:
        freq0=np.linspace(freq.min(),freq.max()/2,freq.size*10)
        f = interp1d(freq, SIGNAL, kind='cubic')
        return freq0, f(freq0)
    else:
        return freq, SIGNAL

def fullspec(data,sr):
    '''
    fullspec (C) aadm 2016-2018
    Calculates amplitude spectrum of 2D numpy array.

    INPUT
    data: 2D numpy array, shape=(traces, samples)
    sr: sample rate in ms

    OUTPUT
    freq: frequency
    amp: amplitude
    db: amplitude in dB scale
    f_peak: average peak frequency
    '''
    amps, peaks = [], []
    for i in range(data.shape[0]):
        trace = data[i,:]
        freq, amp = ampspec(trace,sr)
        peak = freq[np.argmax(amp)]
        amps.append(amp)
        peaks.append(peak)
    amp0 = np.mean(np.dstack(amps), axis=-1)
    amp0 = np.squeeze(amp0)
    db0 = 20 * np.log10(amp0)
    db0 = db0 - np.amax(db0)
    f_peak = np.mean(peaks)
    print('freq peak: {:.2f} Hz'.format(f_peak))
    return freq,amp0,db0,f_peak

def plot_ampspec(freq,amp,f_peak,name=None):
    '''
    plot_ampspec (C) aadm 2016-2018
    Plots amplitude spectrum calculated with fullspec (aageofisica.py).

    INPUT
    freq: frequency
    amp: amplitude
    f_peak: average peak frequency
    '''
    db = 20 * np.log10(amp)
    db = db - np.amax(db)
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5),facecolor='w')
    ax[0].plot(freq, amp, '-k', lw=2)
    ax[0].set_ylabel('Power')
    ax[1].plot(freq, db, '-k', lw=2)
    ax[1].set_ylabel('Power (dB)')
    for aa in ax:
        aa.set_xlabel('Frequency (Hz)')
        aa.set_xlim([0,np.amax(freq)/1.5])
        aa.grid()
        aa.axvline(f_peak, color='r', ls='-')
        if name!=None:
            aa.set_title(name, fontsize=16)

def plot_ampspec2(freq1,amp1,f_peak1,freq2,amp2,f_peak2,name1=None,name2=None):
    '''
    plot_ampspec2 (C) aadm 2016-2018
    Plots overlay of 2 amplitude spectra calculated with fullspec.

    INPUT
    freq1, freq2: frequency
    amp1, amp2: amplitude spectra
    f_peak1, f_peak2: average peak frequency
    '''
    db1 = 20 * np.log10(amp1)
    db1 = db1 - np.amax(db1)
    db2 = 20 * np.log10(amp2)
    db2 = db2 - np.amax(db2)
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5),facecolor='w')
    if name1 is not None:
        label1='{:s} Fp={:.0f} Hz'.format(name1,f_peak1)
        label2='{:s} Fp={:.0f} Hz'.format(name2,f_peak2)
    else:
        label1='Fp={:.0f} Hz'.format(f_peak1)
        label2='Fp={:.0f} Hz'.format(f_peak2)
    ax[0].plot(freq1, amp1, '-k', lw=2, label=label1)
    ax[0].plot(freq2, amp2, '-r', lw=2, label=label2)
    ax[0].fill_between(freq1,0,amp1,lw=0, facecolor='k',alpha=0.25)
    ax[0].fill_between(freq2,0,amp2,lw=0, facecolor='r',alpha=0.25)
    ax[0].set_ylabel('Power')
    ax[1].plot(freq1, db1, '-k', lw=2, label=label1)
    ax[1].plot(freq2, db2, '-r', lw=2,label=label2)
    lower_limit=np.min(ax[1].get_ylim())
    ax[1].fill_between(freq1, db1, lower_limit, lw=0, facecolor='k', alpha=0.25)
    ax[1].fill_between(freq2, db2, lower_limit, lw=0, facecolor='r', alpha=0.25)
    ax[1].set_ylabel('Power (dB)')
    for aa in ax:
        aa.set_xlabel('Frequency (Hz)')
        aa.set_xlim([0,np.amax(freq)/1.5])
        aa.grid()
        aa.axvline(f_peak1, color='k', ls='-')
        aa.axvline(f_peak2, color='r', ls='-')
        aa.legend(fontsize='small')