# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Alexander Daniel Rios

Descripcion:
    Script diseñado para la tarea semanal del curso de PDS, se trata de un
    generador de señales y implementacion de DFT
"""
import numpy as np

def signal_generator( vmax = 1, dc = 0, ff = 1, ph=0, nn = 1, fs = 1, signal='senoidal', over_sampling=1):
    
    ts = 1/fs
    
    N_os = nn*over_sampling
    
    tt_os = np.linspace(0, (nn-1)*ts, N_os).flatten()
    
    x_os = np.array([], dtype=np.float).reshape(N_os,0)
    
    if signal == 'senoidal':
        
        aux = vmax * np.sin(2*np.pi*ff*tt_os + ph) + dc
        x_os = np.hstack([x_os, aux.reshape(N_os,1)] )
        x_os = x_os.reshape(N_os,)
        
    elif signal == 'ruido':
        
        x_os = np.random.random_sample((N_os,))
    
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    x = x_os[::over_sampling]
    return tt_os, x_os, tt, x

def DFT(xx):
    
    N = len(xx)
    n = np.arange(N)
    k = n.reshape(N,1)
    e = np.exp(-2j * np.pi * k * n/N)
   
    x = np.dot(e, xx)
    
    return x


