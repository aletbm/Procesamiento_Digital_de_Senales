import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import tools_psd as tp
from pandas import DataFrame
from IPython.display import HTML

N=1000
fs = 1000
Npad = 10 * N
ts = 1/fs
pi=np.pi
a0 = 2
repeat = 200
tt = np.linspace(0, (N-1)*ts, N).flatten()

tus_resultados = []

ventanas = {1:sg.windows.boxcar, 2:tp.w_Bartlett, 3:tp.w_Hann, 4:tp.w_Blackman, 5:sg.flattop}
w_name = {1:'Rectagular', 2:'Bartlett', 3:'Hann', 4:'Blackman', 5:'Flat-top'}

fr = np.random.uniform(-2, 2, repeat)
omega1 = (pi/2 + fr*((2*pi)/N))*(fs/(2*pi))

plt.close('all')

for i in range(1, 6):
    x1 = a0*np.sin(2*pi*tt.reshape(N,1)*omega1.reshape(1,repeat))
    x1 = x1 * np.array(ventanas[i](N)).reshape(N,1)
    
    ff = np.fft.fftfreq(N, d=1/fs)
    fftx = np.fft.fft(x1, n = N, axis = 0) * (1/N)
    
    ff = ff[0:ff.size//2]
    fftx = fftx[0:ff.size]
    
    a0_ = np.abs(fftx[ff == 250,:]).flatten() 
    a0_ += 2 - np.mean(a0_) 
    
    plt.figure(i, figsize=(15, 7), dpi=100)
    plt.plot(ff, np.abs(fftx))
    plt.xlim(243, 257)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("$|X(\Omega)|$")
    plt.title(f"Ventana {w_name[i]}")
    
    plt.figure(6, figsize=(15, 7), dpi=100)
    plt.hist(a0_, bins=20, label=f"{w_name[i]}")
    plt.title("Histograma de $|X(\Omega_0)|$")
    plt.xlabel("$|X(\Omega_0)|$")
    plt.legend(loc='upper right', shadow=True, fontsize='small')
    
    E = (1/repeat)*np.sum(a0_)
    sesgo = E - a0
    varianza = np.var(a0_)
    
    tus_resultados.append([f'{round(sesgo,4)}', f'{round(varianza,4)}'])
    

df = DataFrame(tus_resultados, columns=['$S_a$', '$V_a$'],
               index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                     ])
HTML(df.to_html())

