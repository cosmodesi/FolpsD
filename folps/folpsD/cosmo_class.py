#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import standard libraries
import numpy as np
from classy import Class


# In[2]:


def run_class(h=0.6711, ombh2=0.022, omch2=0.122, omnuh2=0.0006442, 
              As=2e-9, ns=0.965, z=0.97, z_scale=None, N_ur=2.0328, N_ncdm=1,
              khmin=0.0001, khmax=2.0, nbk=700, 
              w0_fld=None, wa_fld=None, Omkh2=None, deg_ncdm=None, spectra='cb'):
    """
    Generates the linear power spectrum (cb) using CLASS.

    Args:
        h (float): Reduced Hubble constant, H0/100, where H0 is the Hubble constant at z=0.
        ombh2 (float): Physical baryon density, Ω_b h².
        omch2 (float): Physical cold dark matter density, Ω_c h².
        omnuh2 (float): Physical massive neutrino density, Ω_ν h².
        As (float): Amplitude of primordial curvature fluctuations.
        ns (float): Spectral index of the primordial power spectrum.
        z (float): Redshift at which the power spectrum is evaluated.
        z_scale (float or list, optional): Single redshift value or list of redshift values for 
                                           additional scaling (default: None).
        N_ur (float): Number of ultra-relativistic (massless) neutrino species.
        khmin (float): Minimum wave number in units of h/Mpc.
        khmax (float): Maximum wave number in units of h/Mpc.
        nbk (int): Number of k points between khmin and khmax.
        w0_fld (float, optional): Dark energy equation of state parameter w0 (default: None).
        wa_fld (float, optional): Dark energy equation of state evolution parameter wa (default: None). 
                                  If w0_fld is provided but wa_fld is not, wa_fld is set to 0.
        Omkh2 (float, optional): Physical curvature density, Ω_k h² (default: None).
        deg_ncdm (float, optional): Degeneracy of massive neutrino species (default: None).
        spectra (str): Specifies which components to include in the power spectrum (e.g., 'cb' for 
                       cold dark matter + baryons, 'total' for total matter) (default: 'cb').

    Returns:
        kh (numpy.ndarray): Array of wave numbers.
        pk (numpy.ndarray): Array of the linear power spectrum values corresponding to the wave numbers.
        pk_z_scale_list (list, optional): List of arrays of power spectra for each redshift in z_scale, if provided.

    Notes:
        - If w0_fld is specified, the cosmological constant (Omega_Lambda) is set to 0, as we are 
          modeling a dark energy component with a time-varying equation of state.
        - If wa_fld is not specified but w0_fld is, wa_fld defaults to 0, representing no evolution
          in the dark energy equation of state.
    """
    
    params = {
        'output': 'mPk',
        'omega_b': ombh2,
        'omega_cdm': omch2,
        'omega_ncdm': omnuh2, 
        'h': h,
        'A_s': As,
        'n_s': ns,
        'P_k_max_1/Mpc': khmax,
        'z_max_pk':3,             # Default value is 10 
        'N_ur': N_ur,             # Massless neutrinos 
        'N_ncdm': N_ncdm          # Massive neutrinos species
    }
    
    if w0_fld is not None:
        params['Omega_Lambda'] = 0
        params['w0_fld'] = w0_fld
        params['wa_fld'] = 0 if wa_fld is None else wa_fld
    
    if Omkh2 is not None:
        params['Omega_k'] = Omkh2
        
    if deg_ncdm is not None:
        params['deg_ncdm'] = deg_ncdm
        
    try:
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CLASS with error: {e}")
    
    # Specify k [h/Mpc]
    k = np.logspace(np.log10(khmin), np.log10(khmax), num=nbk)
    
    # Compute the linear power spectrum for the provided redshift 'z'
    try:
        if spectra in ['m', 'matter', 'total']:
            Plin = np.array([cosmo.pk_lin(ki * h, z) * h**3 for ki in k])
        else:
            Plin = np.array([cosmo.pk_cb(ki * h, z) * h**3 for ki in k])
    except Exception as e:
        raise RuntimeError(f"Failed to compute power spectrum at z={z} with error: {e}")
        
    #Computing growths: f and D, and sigma8
    fz = cosmo.scale_independent_growth_factor_f(z)
    Dz = cosmo.scale_independent_growth_factor(z)
    s8 = cosmo.sigma(R = 8.0/h, z = z)
    
    fz_scale_values = []
    Dz_scale_values = []
    s8_scale_values = []
    
    # Check if z_scale is a single value or a list
    if z_scale is not None:
        if not isinstance(z_scale, list):
            z_scale = [z_scale]  # Convert to a list if it's a single value
        
        for z_scale_value in z_scale:
            fz_scale_value = cosmo.scale_independent_growth_factor_f(z_scale_value)
            Dz_scale_value = cosmo.scale_independent_growth_factor(z_scale_value)
            s8_scale_value = cosmo.sigma(R=8.0/h, z=z_scale_value)

            fz_scale_values.append(fz_scale_value)
            Dz_scale_values.append(Dz_scale_value)
            s8_scale_values.append(s8_scale_value)
    
    rdrag = cosmo.rs_drag()
    
    return {'k': k, 
            'pk': Plin,
            'fz': fz,
            'Dz': Dz,
            's8': s8,
            'fz_scale': fz_scale_values,
            'Dz_scale': Dz_scale_values,
            's8_scale': s8_scale_values,
            'rbao': rdrag,
            'cosmo': cosmo
            }


# In[ ]:




