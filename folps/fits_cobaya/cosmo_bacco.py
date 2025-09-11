#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")
import baccoemu
emulator = baccoemu.Matter_powerspectrum()


# In[16]:


def run_bacco(h = 0.6711, ombh2 = 0.022, omch2 = 0.122, omnuh2 = 0.0006442, 
              As = 2e-9, ns = 0.965, z = 0.3,
              khmin=0.0001, khmax=2.0, nbk=700):

    Omega_b   = ombh2 / h**2
    Omega_cdm = omch2 / h**2
    Omega_nu  = omnuh2 / h**2
    Omega_cold   = Omega_b + Omega_cdm
    Omega_matter = Omega_cold + Omega_nu

    params_bacco = {
    'omega_cold'    :  Omega_cold,
    'A_s'           :  As, 
    'omega_baryon'  :  Omega_b,
    'ns'            :  ns,
    'hubble'        :  h,
    'neutrino_mass' :  omnuh2 * 93.14, 
    'w0'            : -1.0,
    'wa'            :  0.0,
    'expfactor'     :  1/(1+z),
    }
    
    k = np.logspace(np.log10(khmin), np.log10(khmax), num=nbk)
    
    k_bacco, plin_bacco = emulator.get_linear_pk(k=k, cold=True, **params_bacco)

    sigma8_cold = emulator.get_sigma8(**params_bacco, cold=False)

    #print(sigma8_cold)
    
    return {'k': k_bacco, 
            'pk': plin_bacco,
            'sigma8': sigma8_cold,
            'Omega_m':Omega_matter
            }


# In[ ]:




