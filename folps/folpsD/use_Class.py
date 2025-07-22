#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import sys, platform, os, subprocess
from classy import Class


def generate_ps(h = 0.6711, ombh2 = 0.022, omch2 = 0.122, omnuh2 = 0.0006442, 
                As = 2e-9, ns = 0.965, z = 0.97, z_scale = [0.61, 1.4], N_ur = 2.0328,
                khmin = 0.0001, khmax = 2.0, nbk = 1000):
    '''Generates the linear (cb) power spectrum using Class.
    
    Args:
        h = H0/100, with H0 the Hubble constant,
        omXh2: Omega_X h², where X = b (baryons), c (CDM), nu (neutrinos),
        As: amplitude of primordial curvature fluctuations,
        ns: spectral index,
        z: redshift,
        z_scale: array of redshift to scale the linear power spectrum, and non-linear terms.
        khmin, khmax: minimal and maximal wave-number,
        nbk: number of points in [khmin, khmax].
        
    Rertuns:
        kh: vector of wave-number,
        pk: linear (cb) power spectrum,
    '''
    
    params = {
             'output':'mPk',
             'omega_b':ombh2,
             'omega_cdm':omch2,
             'omega_ncdm':omnuh2, 
             'h':h,
             'A_s':As,
             'n_s':ns,
             'P_k_max_1/Mpc':khmax,
             'z_max_pk':10.,         #Default value is 10 
             #'N_eff':Neff,
             'N_ur':N_ur,            #massless neutrinos 
             'N_ncdm':1              #massive neutrinos species
             }
    
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    #Specify k
    k = np.logspace(np.log10(khmin*h), np.log10(khmax*h), num = nbk) #Mpc^-1
    
    #Computes the linear (cb) power spectrum
    Plin = np.array([cosmo.pk_cb(ki, z) for ki in k])
    
    #Tranforming to h/Mpc and (Mpc/h)^3
    k /= h
    Plin *= h**3
    
    #Computing f and D
    fz = cosmo.scale_independent_growth_factor_f(z)
    Dz = cosmo.scale_independent_growth_factor(z)
    
    
    fz_scale_values = []
    Dz_scale_values = []

    for z_scale_value in z_scale:
        fz_scale_value = cosmo.scale_independent_growth_factor_f(z_scale_value)
        Dz_scale_value = cosmo.scale_independent_growth_factor(z_scale_value)

        fz_scale_values.append(fz_scale_value)
        Dz_scale_values.append(Dz_scale_value)
    
    
    return({'kh':k, 'pk':Plin,
            'fz':fz, 
            'fz_scale': fz_scale_values,
            'Dz':Dz, 
            'Dz_scale': Dz_scale_values, 
            'cosmo':cosmo})


# In[ ]:




