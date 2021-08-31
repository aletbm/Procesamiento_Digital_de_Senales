from tools_psd import signal_generator, DFT
import numpy as np
import matplotlib.pyplot as plt
#%%
fig1 = plt.figure(1, figsize=(10, 6), dpi=100)

fig1.add_subplot(2,2,(1, 2))

ff = 200
fs = 1000
nn = 1000
vmax = 5
dc = 0
ph = (np.pi)/2
tipo_sg = 'senoidal'

tt1, xx1 = signal_generator(vmax, dc, ff, ph, nn, fs, tipo_sg)

line_hdls = plt.plot(tt1, xx1, label=f"Vmax: {vmax}v\nVmed: {dc}v\nFrecuencia: {ff}Hz\nFase: {ph}rad")
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.title('Generador de Señales')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.legend(loc='upper right', shadow=True, fontsize='small')

#%%
fig1.add_subplot(2,2,3)

x_dft = DFT(xx1)

N = len(x_dft) 
n=np.arange(N)   
T = N/fs
freq = n/T

line_hdls = plt.plot(freq, abs(x_dft), label=f"Frecuencia: {ff}Hz")
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.title('Transformada Discreta de Fourier (DFT)')
plt.xlabel('frecuencia [Hz]')
plt.ylabel('DFT-Amplitud [|H(freq)|]')
plt.legend(loc='upper right', shadow=True, fontsize='small')

#%%
fig1.add_subplot(2,2,4)

x_fft = np.fft.fft(xx1)

line_hdls = plt.plot(freq, abs(x_dft), label=f"Frecuencia: {ff}Hz")
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.title('Transformada Rapida de Fourier (FFT)')
plt.xlabel('frecuencia [Hz]')
plt.ylabel('FFT-Amplitud [|H(freq)|]')
plt.legend(loc='upper right', shadow=True, fontsize='small')

plt.tight_layout(h_pad=1.0)

#%%
fig2 = plt.figure(2, figsize=(10, 6), dpi=100)
fig2.add_subplot(2,1,1)

tt2, xx2 = signal_generator(nn=nn, fs=fs, signal='ruido')

line_hdls = plt.plot(tt2, xx2, label="Vmax: 1v")
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.title('Generador de Señales' )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.legend(loc='upper right', shadow=True, fontsize='small')

fig2.add_subplot(2,1,2)

x2_dft = DFT(xx2)

N = len(x_dft) 
n=np.arange(N)   
T = N/fs
freq2 = n/T

line_hdls = plt.plot(freq2, abs(x2_dft), label=f"Frecuencia: {ff}Hz")
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.title('Transformada Rapida de Fourier (FFT)')
plt.xlabel('frecuencia [Hz]')
plt.ylabel('FFT-Amplitud [|H(freq)|]')
plt.legend(loc='upper right', shadow=True, fontsize='small')

plt.tight_layout(h_pad=2.0)
#%%
plt.show()
