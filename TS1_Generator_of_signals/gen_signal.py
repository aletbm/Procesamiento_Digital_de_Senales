# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Alexander Daniel Rios

Descripcion:
    Script diseñado para la tarea semanal del curso de PDS, se trata de un
    generador de señales
"""
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = 1, fs = 1):
    ts = 1/fs
    
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    
    x = np.array([], dtype=np.float).reshape(nn,0)
    
    aux = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    
    x = np.hstack([x, aux.reshape(nn,1)] )
    
    plt.figure(1)
    
    line_hdls = plt.plot(tt, x, label=f"Vmax: {vmax}v\nVmed: {dc}v\nFrecuencia: {ff}Hz\nFase: {ph}rad")
    
    plt.title('Generador de Señales' )
    
    plt.xlabel('tiempo [segundos]')
    
    plt.ylabel('Amplitud [V]')
    
    plt.legend(loc='upper right', shadow=True, fontsize='small')
    
    #axes_hdl = plt.gca()

    #axes_hdl.legend(line_hdls, 'frecuencia: ' + str(ff), loc='upper right')
    
    plt.show()
    
    
    
mi_funcion_sen(4, 4, 10, np.pi, 1000, 1000)