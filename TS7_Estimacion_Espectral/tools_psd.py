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
    
    x_os = np.array([], dtype=np.float64).reshape(N_os,0)
    
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

def w_Bartlett(M):
    N = M - 1
    w = np.linspace(0, N, M)
    
    w[0:N//2] = 2 * w[0:N//2]/N
    w[N//2:] = 2 - (2 * w[N//2:]/N)
    return  w

def w_Hann(M):
    N = M - 1
    w = np.linspace(0, N, M)
    
    w[0:N+1] = 0.5 - 0.5*np.cos((2*np.pi*w[0:N+1])/N)
    return  w

def w_Blackman(M):
    N = M - 1
    w = np.linspace(0, N, M)
    
    w[0:N+1] = 0.42 - 0.5*np.cos((2*np.pi*w[0:N+1])/N) + 0.08*np.cos((4*np.pi*w[0:N+1])/N)
    return  w

def first_zero(fftw, step):
    min_backup = fftw[0:step].argmin()
    encontrado = 0
    i = 1
    while(encontrado == 0):
        i += 1
        min_ = fftw[0:step*i].argmin()    
        if min_ == min_backup:
            encontrado = 1
        else:
            min_backup = min_
    return min_

def second_max(fftw, step, init):
    max_backup = fftw[init:init+step].argmax()
    encontrado = 0
    i = 1
    while(encontrado == 0):
        i += 1
        max_ = fftw[init:init+(step*i)].argmax()    
        if max_ == max_backup:
            encontrado = 1
        else:
            max_backup = max_
    return max_+init


