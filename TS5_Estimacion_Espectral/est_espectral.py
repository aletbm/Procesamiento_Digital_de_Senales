from tools_psd import signal_generator
import numpy as np
import matplotlib.pyplot as plt

fs = 1000
nn = 1000
N = 1 + 9
nn_pad = N*nn
ko = nn_pad/4
ff = [ko*(fs/nn_pad), ko*(fs/nn_pad) + 0.25 ,ko*(fs/nn_pad) + 0.5]
vmax = 1
dc = 0
ph1 = 0
tipo_sg = 'senoidal'
ov_s=1

s=[None]*len(ff)
s2=[None]*len(ff)
sn=[None]*len(ff)
x_fft=[]
f=[]
colores = ['red', 'blue', 'orange', 'green', 'silver', 'yellow', 'gray', 'violet', 'pink', 'brown', 'purple', 'gold']

#=====================================Señal================
for i in range(0, len(ff)):
    tt_os, xx_os, tsig, s[i] = signal_generator(vmax, dc, ff[i], ph1, nn, fs, tipo_sg, over_sampling=ov_s)
    
    pot_S = np.sum(s[i]**2)*(1/nn)          #Potencia de la señal
    s[i] = s[i]/np.sqrt(pot_S)              #Señal normalizada en potencia Psn = 1
    
    s2[i] = s[i].copy()                     #Copia superficial
    s2[i].resize(nn_pad)                    #Resize para generar el zero padding
    
    sn[i] = s2[i]*(nn_pad/nn)              #Señal normalizada por el kernel de Dirichtled
    
    x_fft.append(np.fft.fft(sn[i].reshape(nn_pad), axis=0)*( 1 / nn_pad))   #Transformada discreta de Fourier de S
    f.append(np.fft.fftfreq(nn_pad, d=1/fs))    # Eje de frecuencias
    
    pot_dep = sum(np.abs(x_fft[i])**2)
    
    ######## para F>0 #######
    f[i] = f[i][:f[i].size//2]
    x_fft[i] = x_fft[i][:x_fft[i].size//2]
    
    ########################################
    plt.figure(1, figsize=(10, 7), dpi=100)
    plt.plot(f[i], 10*np.log10(2*np.abs(x_fft[i])**2),'x:', color=colores[i], label=f'Fs={ff[i]} Hz\n  Pot_PSD={pot_dep:0.2f}')
    plt.ylim(-100, 2)
    plt.xlim(0,500)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    
    plt.figure(2, figsize=(10, 7), dpi=100)
    plt.plot(f[i], 10*np.log10(2*np.abs(x_fft[i])**2),'x:', color=colores[i], label=f'Fs={ff[i]} Hz')
    plt.xlim((fs/4)-1, (fs/4)+1)
    plt.ylim(-10, 1)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    
    plt.figure(3, figsize=(10, 7), dpi=100)
    plt.plot(f[i], 10*np.log10(2*np.abs(x_fft[i])**2),'x:', color=colores[i], label=f'Fs={ff[i]} Hz')
    plt.xlim((fs/4)-1, (fs/4)+1)
    plt.ylim(-0.1, 0.01)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    
    plt.rcParams.update({'font.size': 8})