"""Scientific wave plots from one saved diagnostic trajectory; never training features."""
import pathlib,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
output=ROOT/'results/aggregate'
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for row,variant in enumerate(('fixed','trained')):
    record=np.load(output/f'waveforms-{variant}.npz')
    mode=record['mode'];dt=float(record['token_duration_s']);dx=float(record['cell_spacing_m'])
    im=axes[row,0].imshow(abs(mode).T,aspect='auto',origin='lower',extent=[0,len(mode)*dt*1e9,0,mode.shape[1]*dx*1e6],cmap='magma')
    axes[row,0].set_ylabel('Periodic coordinate [micrometre]');axes[row,0].set_title(f'{variant}: calculated mode amplitude')
    fig.colorbar(im,ax=axes[row,0],label='Envelope amplitude')
    t=np.arange(len(mode))*dt*1e9
    axes[row,1].plot(t,mode[:,8].real,label='Re mode at cell 8')
    axes[row,1].plot(t,mode[:,8].imag,label='Im mode at cell 8')
    axes[row,1].plot(t,record['observation'][:,0]/100,label='Receiver I, first port',alpha=.75)
    axes[row,1].set_title(f'{variant}: state versus measured receiver');axes[row,1].legend(fontsize=8)
for ax in axes.flat:ax.set_xlabel('Time [ns]')
fig.savefig(output/'mode_waveforms.png',dpi=180)
