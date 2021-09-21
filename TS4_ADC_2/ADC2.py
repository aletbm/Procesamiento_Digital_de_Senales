from tools_psd import signal_generator
import numpy as np
import matplotlib.pyplot as plt

#%%

def cuantizador(signal, vf, B):
    q=vf/2**B
    sr=np.round(signal/q)
    sr = sr*q
    for i,m in enumerate(sr):
        if m > vf:
            sr[i] = vf
        elif m < -vf:
            sr[i] = -vf
    return sr, q

#plt.close()
fig1 = plt.figure(1, figsize=(10, 10), dpi=100)

#========Generador=============
ff = 1
fs = 1000
nn = 1000
vmax = 1
dc = 0
ph1 = 0
tipo_sg = 'senoidal'
ov_s=100
#==============================

#=======Datos ADC==============
vf=2  #Rango en volts del ADC  
B=4   #Resolucion del ADC
q=vf/2**B
#==============================

#=========Señal================
tt_os, xx_os, tt1, s = signal_generator(vmax, dc, ff, ph1, nn, fs, tipo_sg, over_sampling=ov_s)
s = s/np.sqrt(np.var(s))
#==============================

#=======Ruido analogico========
Kn = 1
pot_n = ((q**2)/12)*Kn
r = np.random.normal(0, np.sqrt(pot_n), nn)
r = r.reshape(nn,)
sr = np.add(s, r)
#==============================

#===========ADC================
sq, q = cuantizador(sr, vf, B)
#==============================

fig1.add_subplot(3,1,1)
plt.plot(tt1, s, linewidth=0.8, color='red')
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.xlabel('Tiempo[s]')
plt.ylabel('Señal de Entrada(SE)[V]')
plt.title(f'Señal muestreada por un ADC de {B} bits - $\pm V_R= $ {vf} V - q = {q} V')

fig1.add_subplot(3,1,2)
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.xlabel('Tiempo[s]')
plt.ylabel('SE + Ruido(SER)[V]')
plt.plot(tt1, sr, linewidth=0.3, color='red')

fig1.add_subplot(3,1,3)
plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
plt.xlabel('Tiempo[s]')
plt.ylabel('SER Cuantizada [V]')
plt.plot(tt1, sq, linewidth=0.3)
# markerline, stemline, baseline = plt.stem(tt1, sr, basefmt=" ")
# plt.setp(stemline, linewidth = 0.3)
# plt.setp(markerline, markersize = 0.7)

#%%
err = sr - sq

fig2 = plt.figure(2, figsize=(10, 10), dpi=100)
fig2.add_subplot(2,1,1)
plt.plot(tt1, err, linewidth=0.3)
plt.xlabel('Tiempo[s]')
plt.ylabel('Amplitud[V]')
plt.title(f'Error de cuantizacion para {B} bits - $\pm V_R= $ {vf} V - q = {q} V')

fig2.add_subplot(2,1,2)
plt.hist(err, bins=10, label=f"$\sigma^2$={np.var(err)}\n media={np.mean(err)}")
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, nn/10, nn/10, 0]), '--r' )
plt.xlabel('Tension[V]')
plt.ylabel('Cantidad de veces')
plt.legend(loc='upper right', shadow=True, fontsize='small')


#%%
acorr = np.correlate(err, err, mode='full')*(1/nn)
acorr = acorr[acorr.size // 2 : ]

f = np.fft.fftfreq(nn, d=1/fs)

dep = np.fft.fft(err)/nn

pot_total = sum(np.abs(dep)**2)

plt.figure(3, figsize=(10, 5), dpi=100)
markerline, stemline, baseline = plt.stem(tt1, acorr, basefmt=" ", label=f'Acorr en el origen={acorr[0]}')
plt.setp(stemline, linewidth = 0.3)
plt.setp(markerline, markersize = 0.7)
plt.title('Autocorrelacion del Ruido Analogico')
plt.xlabel('Tiempo[s]')
plt.ylabel('Autocorrelacion')
plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.title(f'Autocorrelacion del Error de cuantizacion para {B} bits - $\pm V_R= $ {vf} V - q = {q} V')

plt.figure(4, figsize=(10, 5), dpi=100)
markerline, stemline, baseline = plt.stem(f, (np.abs(dep))**2, basefmt=" ", label=f'Potencia total={pot_total}')
plt.setp(stemline, linewidth = 0.3)
plt.setp(markerline, markersize = 0.7)
plt.title('Densidad Espectral de Potencia del Error de cuantizacion')
plt.xlabel('Frecuencia[Hz]')
plt.ylabel('Potencia/Frecuencia[Watts/Hz]')
plt.legend(loc='upper right', shadow=True, fontsize='small')

#%%
plt.figure(5, figsize=(10, 5), dpi=100)

f = np.fft.fftfreq(nn, d=1/fs)

ft_R = (1/nn)*np.fft.fft(r)
ft_S = (1/nn)*np.fft.fft(s)
ft_SQ = (1/nn)*np.fft.fft(sq)
ft_SR = (1/nn)*np.fft.fft(sr)
ft_ERR = (1/nn)*np.fft.fft(err)

ff = f[:f.size // 2]

mean_ERR = np.mean(np.abs(ft_ERR)**2)
mean_R = np.mean(np.abs(ft_R)**2)

plt.plot(ff, 10*np.log10(2*np.abs(ft_SQ[:ft_SQ.size // 2])**2), linewidth=1, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(ff, 10*np.log10(2*np.abs(ft_S[:ft_S.size // 2])**2), color='orange', ls='dotted', linewidth=0.7, label='$ s $ (analog)')
plt.plot(ff, 10*np.log10(2*np.abs(ft_SR[:ft_SR.size // 2])**2), ':g', linewidth=0.7, label='$ s_R = s + n $  (ADC in)')

plt.plot(ff, 10*np.log10(2*np.abs(ft_R[:ft_R.size // 2])**2), ':r', linewidth=0.7)
plt.plot(np.arange(nn//2), 10*np.log10(2*(np.zeros(nn//2)+mean_R)),':r', linewidth=0.7, label= '$ \overline{n} = $'+f'{round(10*np.log10(2*mean_R),2)} dB (piso analog.)')

plt.plot(ff, 10*np.log10(2*np.abs(ft_ERR[:ft_ERR.size // 2])**2),':c', linewidth=0.7)
plt.plot(np.arange(nn//2), 10*np.log10(2*(np.zeros(nn//2)+mean_ERR)),':c', linewidth=0.7, label='$ \overline{n_Q} = $' + f'{round(10*np.log10(2*mean_ERR),2)} dB (piso digital)')

plt.legend(loc='center right', shadow=True, fontsize='small')
plt.xlabel('Frecuencia[Hz]')
plt.ylabel('Densidad de Potencia[dB]')
plt.title(f'Señal muestreada por un ADC de {B} bits - $\pm V_R= $ {vf} V - q = {q} V')








    