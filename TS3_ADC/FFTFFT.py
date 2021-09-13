from tools_psd import signal_generator
import numpy as np
import matplotlib.pyplot as plt

#%%

def cuantizador(signal, vf, B):
    q=vf/(2**(B-1))
    sr=np.round(signal/q)
    sr = sr*q
    for i,m in enumerate(sr):
        if m > vf:
            sr[i] = vf
        elif m < -vf:
            sr[i] = -vf
    return sr, q

fig1 = plt.figure(1, figsize=(10, 10), dpi=100)

#========Generador=============
ff = 1
fs = 10000
nn = 10000
vmax = 1
dc = 0
ph1 = 0
tipo_sg = 'senoidal'
ov_s=10
#==============================

#=======Datos ADC==============
vf=2  #Rango en volts del ADC  
B=4   #Resolucion del ADC
q=vf/(2**(B-1))
#==============================

#=========Señal================
tt_os, xx_os, tt1, xx1 = signal_generator(vmax, dc, ff, ph1, nn, fs, tipo_sg, over_sampling=ov_s)
#==============================

#=======Ruido analogico========
r = np.random.normal(0, q/2, nn)
r = r.reshape(nn,)
xx_r = np.add(xx1, r)
#==============================

#===========ADC================
sr, q = cuantizador(xx_r, vf, B)
#==============================

fig1.add_subplot(3,1,1)
plt.plot(tt1, xx1, linewidth=0.8, color='red')
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.xlabel('Tiempo[s]')
plt.ylabel('Señal de Entrada(SE)[V]')
plt.title(f'Simulacion ADC - Res.:{B} Bits, q={q}')

fig1.add_subplot(3,1,2)
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.xlabel('Tiempo[s]')
plt.ylabel('SE + Ruido(SER)[V]')
plt.plot(tt1, xx_r, linewidth=0.3, color='red')

fig1.add_subplot(3,1,3)
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.xlabel('Tiempo[s]')
plt.ylabel('SER Cuantizada [V]')
plt.plot(tt1, sr, linewidth=0.3)
# markerline, stemline, baseline = plt.stem(tt1, sr, basefmt=" ")
# plt.setp(stemline, linewidth = 0.3)
# plt.setp(markerline, markersize = 0.7)

#%%
err = xx_r - sr

fig2 = plt.figure(2, figsize=(10, 10), dpi=100)
fig2.add_subplot(2,1,1)
plt.plot(tt1, err)
plt.title('Error de cuantizacion')

fig2.add_subplot(2,1,2)
plt.hist(err, bins=10)

#%%
plt.figure(3)

dep = np.fft.fft(err)/nn
acorr = np.correlate(err, err, mode='full')*(1/nn)
acorr = acorr[acorr.size // 2 : ]
markerline, stemline, baseline = plt.stem(tt1, acorr, basefmt=" ")
plt.setp(stemline, linewidth = 0.3)
plt.setp(markerline, markersize = 0.7)

plt.figure(4)

f = np.fft.fftfreq(nn, d=1/fs)

markerline, stemline, baseline = plt.stem(f, (np.abs(dep))**2, basefmt=" ")
plt.setp(stemline, linewidth = 0.3)
plt.setp(markerline, markersize = 0.7)

pot_total = sum(np.abs(dep)**2)
print(pot_total)
print(acorr[0])

#%%
plt.figure(5)

plt.plot(f, 20*np.log10(np.abs(np.fft.fft(sr))**2), linewidth=0.7)
plt.xlim(0, fs/2)
# #%%
# #==========FFT===============
# fig3 = plt.figure(4, figsize=(10, 10), dpi=100)
# n=np.arange(nn)   
# T = nn/fs
# freq = n/T

# x_fft = np.fft.fft(xx1.reshape(nn))*( 2 / nn)

# fig3.add_subplot(2,1,1)
# plt.stem(freq, abs(x_fft), basefmt=" ")
# plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
# plt.xlim([0, fs/2])
# plt.xlabel('frecuencia[Hz]')
# plt.ylabel('|H(z)|')

# fig3.add_subplot(2,1,2)
# plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
# plt.stem(freq, np.angle(x_fft), basefmt=" ")
# plt.xlim([0, fs/2])
# plt.xlabel('frecuencia[Hz]')
# plt.ylabel('angle(H(z))')

# plt.show()







    