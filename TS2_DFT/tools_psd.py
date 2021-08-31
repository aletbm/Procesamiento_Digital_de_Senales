# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Alexander Daniel Rios

Descripcion:
    Script diseñado para la tarea semanal del curso de PDS, se trata de un
    generador de señales y implementacion de DFT
"""
import numpy as np

def signal_generator( vmax = 1, dc = 0, ff = 1, ph=0, nn = 1, fs = 1, signal='senoidal'):
    
    ts = 1/fs
    
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    
    x = np.array([], dtype=np.float).reshape(nn,0)
    
    if signal == 'senoidal':
        
        aux = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
        x = np.hstack([x, aux.reshape(nn,1)] )
        
    elif signal == 'ruido':
        x = np.random.random_sample((nn,))
    
    return tt, x

def DFT(xx):
    
    N = len(xx)
    n = np.arange(N)
    k = n.reshape(N,1)
    e = np.exp(-2j * np.pi * k * n/N)
   
    x = np.dot(e, xx)
    
    return x


