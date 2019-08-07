import segyio, numpy as np
import matplotlib.pyplot as plt

# print(header)
def read_segy(f, lag=0, twod=False, byte_il=189, byte_xl=193):
    '''
    read_segy (C) aadm 2018 // using Statoil's segyio
    '''
    if twod:
        with segyio.open(f, 'r', ignore_geometry=True) as segyfile:
            sr = segyio.tools.dt(segyfile)/1e3
            nsamples = segyfile.samples.size
            twt = segyfile.samples
            ntraces = segyfile.tracecount
            data = segyfile.trace.raw[:]
            header = segyio.tools.wrap(segyfile.text[0])      
    else:
        with segyio.open(f, iline=byte_il, xline=byte_xl) as segyfile:
            sr = segyio.tools.dt(segyfile)/1e3
            nsamples = segyfile.samples.size
            twt = segyfile.samples
            ntraces = segyfile.tracecount
            data = segyio.tools.cube(segyfile)
            header = segyio.tools.wrap(segyfile.text[0])  
            inlines = segyfile.ilines
            crosslines = segyfile.xlines
    size_mb= data.nbytes/1024**2
    print('[read_segy] number of traces: {0}, samples: {1}, sample rate: {2} s'.format(ntraces,nsamples,sr))
    print('[read_segy] first, last sample twt: {0}, {1} s'.format(twt[0],twt[-1]))
    print('[read_segy] size: {:.2f} Mb ({:.2f} Gb)'.format(size_mb, size_mb/1024))
    if not twod:
        print('[read_segy] inlines: {:.0f}, min={:.0f}, max={:.0f}'.format(inlines.size,inlines.min(),inlines.max()))
        print('[read_segy] crosslines: {:.0f}, min={:.0f}, max={:.0f}'.format(crosslines.size,crosslines.min(),crosslines.max()))
        return data, nsamples, sr, twt, ntraces, header, inlines, crosslines
    else:
        return data, nsamples, sr, twt, ntraces, header
def plot_seismic(inputseis,twt,name,colr='seismic',clip_val=None):
    ntraces=np.shape(inputseis)[1]
    if clip_val is None:
        clip_val=abs(np.percentile(inputseis, 0.999))   # default clip value
    f, ax = plt.subplots(figsize=(16,6))
    im=ax.imshow(inputseis,interpolation='bilinear',aspect='auto',cmap=colr,extent=(0,ntraces,twt[-1],twt[0]),vmin=-clip_val,vmax=clip_val)
    plt.xlabel('Trace no.'), plt.ylabel('Two-way time [ms]')
    plt.title(name), plt.grid(), plt.colorbar(im)