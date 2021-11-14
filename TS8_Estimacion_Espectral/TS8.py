import numpy as np
import matplotlib.pyplot as plt
from spectrum import WelchPeriodogram, pcovar, pcorrelogram
from pandas import DataFrame

pi = np.pi
fs = 1000
padding = 10
N = 1000
repeat = 200

t = np.linspace(0, (N-1)/fs, N)
f = np.linspace(0, (N-1)*(fs/N), N)

fr = np.random.uniform(-1/2, 1/2, repeat)
omega0 = pi/2
omega1 = (omega0 + fr*((2*pi)/N))*(fs/(2*pi))

pot = np.array([3,10])

a1 = 1
noise_p = a1/(10**(pot/10))
SNR = 10 * np.log10(a1/noise_p)

#%%
noise = np.random.normal(0, np.sqrt(noise_p[0]), (N, repeat))
senal = (np.sqrt(a1*2))*np.sin(2*pi*t.reshape(N, 1)*omega1.reshape(1, repeat))
x1 =  senal + noise

Pgram = (1/N)*np.abs(np.fft.fft(x1, n=N*padding, axis = 0))**2

plt.close('all')
plt.figure(1, figsize=(15, 7), dpi=100)
pk_ar1 = np.empty(repeat)
for i in range(0,repeat):
    pp, r = WelchPeriodogram(x1[:, i], NFFT=padding*N, sampling=fs)#, lag=3*N//4
    pp_welch = pp[0]
    f_welch = pp[1]
    peaks = np.argmax(pp_welch, axis = 0)
    peaks_n = pk_ar1[i] = peaks/padding
    plt.plot(peaks_n, 10*np.log10(pp_welch[peaks]), "X")
    plt.title('Welch - SNR:3dB')

pk_ar2 = np.empty(repeat)
plt.figure(2, figsize=(15, 7), dpi=100)
for i in range(0,repeat):
    p = pcovar(x1[:,i], order=15, NFFT=padding*N, sampling=fs)
    d = p.psd
    peaks = np.argmax(d, axis = 0)
    peaks_n = pk_ar2[i] = peaks/padding
    plt.plot(peaks_n, 10*np.log10(d[peaks]/d[peaks].max()), "X")
    p(); p.plot(norm=True)
    plt.title('Covar(15) - SNR:3dB')

pk_ar3 = np.empty(repeat)
plt.figure(3, figsize=(15, 7), dpi=100)
old = np.seterr(invalid='ignore') # pcorrelogram preseenta un warning debido a log10, con esto no nos imprime el warning

for i in range(0,repeat):
    p = pcorrelogram(x1[:,i], lag=15, NFFT=padding*N, sampling=fs)
    d = p.psd
    peaks = np.argmax(d, axis = 0)
    peaks_n = pk_ar3[i] = peaks/padding
    plt.plot(peaks_n, 10*np.log10(d[peaks]/d[peaks].max()), "X")
    p(); p.plot(norm=True)
    plt.title('Correlograma - SNR:3dB')
    
np.seterr(**old) #volvemos a activar los warnings por valor invalidos

errorw3 = pk_ar1 - omega1
errorc3 = pk_ar2 - omega1
errorb3 = pk_ar3 - omega1

#%%
headers = ["Periodograma", "SNR", "Valor Medio", "Varianza"]
per = ["Welch", "Covar", "Correlograma"]
print(f'{headers[0]:^15s}|{headers[1]:^15s}|{headers[2]:^15s}|{headers[3]:^15s}')
print('------------------------------------------------------------')
print(f'{per[0]:^15s}|{SNR[0]:^15f}|{np.mean(errorw3):^15f}|{np.var(errorw3):^15f}')
print(f'{per[1]:^15s}|{SNR[0]:^15f}|{np.mean(errorc3):^15f}|{np.var(errorc3):^15f}')
print(f'{per[2]:^15s}|{SNR[0]:^15f}|{np.mean(errorb3):^15f}|{np.var(errorb3):^15f}')

#%%
noise = np.random.normal(0, np.sqrt(noise_p[1]), (N, repeat))
senal = (np.sqrt(a1*2))*np.sin(2*pi*t.reshape(N, 1)*omega1.reshape(1, repeat))
x1 =  senal + noise

plt.figure(4, figsize=(15, 7), dpi=100)
pk_ar1 = np.empty(repeat)
for i in range(0,repeat):
    pp, r = WelchPeriodogram(x1[:, i], NFFT=padding*N, sampling=fs)
    pp_welch = pp[0]
    f_welch = pp[1]
    peaks = np.argmax(pp_welch, axis = 0)
    peaks_n = pk_ar1[i] = peaks/padding
    plt.plot(peaks_n, 10*np.log10(pp_welch[peaks]), "X")
    plt.title('Welch - SNR:10dB')

pk_ar2 = np.empty(repeat)
plt.figure(5, figsize=(15, 7), dpi=100)
for i in range(0,repeat):
    p = pcovar(x1[:,i], order=15, NFFT=padding*N, sampling=fs)
    d = p.psd
    peaks = np.argmax(d, axis = 0)
    peaks_n = pk_ar2[i] = peaks/padding
    plt.plot(peaks_n, 10*np.log10(d[peaks]/d[peaks].max()), "X")
    p(); p.plot(norm=True) 
    plt.title('Covar(15) - SNR:10dB')

pk_ar3 = np.empty(repeat)
plt.figure(6, figsize=(15, 7), dpi=100)
old = np.seterr(invalid='ignore') # pcorrelogram preseenta un warning debido a log10, con esto no nos imprime el warning

for i in range(0,repeat):
    p = pcorrelogram(x1[:,i], lag=15, NFFT=padding*N, sampling=fs)
    d = p.psd
    peaks = np.argmax(d, axis = 0)
    peaks_n = pk_ar3[i] = peaks/padding
    plt.plot(peaks_n, 10*np.log10(d[peaks]/d[peaks].max()), "X")
    p(); p.plot(norm=True)
    plt.title('Correlograma - SNR:10dB')
    
np.seterr(**old) #volvemos a activar los warnings por valor invalidos

errorw10 = pk_ar1 - omega1
errorc10 = pk_ar2 - omega1
errorb10 = pk_ar3 - omega1
print('------------------------------------------------------------')
print(f'{per[0]:^15s}|{SNR[1]:^15f}|{np.mean(errorw10):^15f}|{np.var(errorw10):^15f}')
print(f'{per[1]:^15s}|{SNR[1]:^15f}|{np.mean(errorc10):^15f}|{np.var(errorc10):^15f}')
print(f'{per[2]:^15s}|{SNR[1]:^15f}|{np.mean(errorb10):^15f}|{np.var(errorb10):^15f}')

# fs = 1000
# N = np.array([10, 50, 100, 250, 500, 1000, 5000])
# sigma2 = 2

# sesgo = np.empty(len(N))
# varianza = np.empty(len(N))
# area = np.empty(shape=(1,200))

# m = 200

# plt.close("all")

# for i in range(0, len(N)):
#     x = np.random.normal(0, np.sqrt(sigma2), (N[i], m))

#     Pgram = (1/N[i])*np.abs(np.fft.fft(x, axis = 0))**2

#     E_Pgram = np.mean(Pgram, axis=1)
#     area = np.mean(E_Pgram)
#     sesgo[i] = area - sigma2
#     varianza[i] = np.mean(np.var(Pgram, axis=1))

# t = np.linspace(0, (N[i]-1)/fs, N[i])
# f = np.linspace(0, (N[i]-1)*(fs/N[i]), N[i])
# plt.figure(i+1)
# plt.plot(f, E_Pgram)
# plt.figure(i+8)
#plt.hist(varianza, bins=30)


    