#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cosmodesi-main
#     language: python
#     name: cosmodesi-main
# ---

# +
import FOLPSD as FOLPS
#from use_Class import * 
from cosmo_bacco import *

import sys, os, shutil
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
# from datetime import datetime
from mike_data_tools import *

derived_params = ['sigma8', 'Omega_m']

k_min=0.02
k_max=0.301
k_max_b0 = 0.14
k_max_b2 = 0.10

isP0, isP2, isP4 =True, True, False
isB000, isB202 = False, False

Vol=1

tracer='LRG'
z_str='z0.800'
z_evaluation=0.8

path_fits='chains/'

# now = datetime.now()
# tiempo=now.strftime("%m-%d-%Y-%H%M")

name=f"c_FolpsD_{tracer}_z{z_evaluation:.3f}_Pkmax-{k_max:.3f}_bsfree"


chains_filename = path_fits+name+".h5"
copy_filename = path_fits+name+".py"
print(chains_filename)


# +
k_data_2nd,pkl0_2nd_all,pkl2_2nd_all,a,B000_2nd_all,B202_2nd_all = ExtractDataAbacusSummit(tracer,z_str,
                                                                                           subtract_shot=True)

Pk_0_2nd = np.mean(pkl0_2nd_all,axis = 0)
Pk_2_2nd = np.mean(pkl2_2nd_all,axis = 0)
B000_2nd = np.mean(B000_2nd_all,axis = 0)
B202_2nd = np.mean(B202_2nd_all,axis = 0)


# +
def is_python_script():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Running in a Jupyter notebook or qtconsole
            return False
        elif shell == 'TerminalInteractiveShell':
            # Running in IPython terminal
            return True
        else:
            # Other types of shells
            return True
    except NameError:
        # Probably running as a standard Python script
        return True

def copy_and_rename_if_py(new_filename):
    # Get the current file name
    current_file = os.path.realpath(__file__)

    # Get the absolute path of the directory containing the current file
    target_directory = os.path.dirname(current_file)

    # Only proceed if it's a .py file
    if current_file.endswith('.py'):
        # Ensure the target directory exists (should always exist, but safe to check)
        os.makedirs(target_directory, exist_ok=True)

        # Full destination path with the new file name
        destination = os.path.join(target_directory, new_filename)

        # Copy and rename the file
        shutil.copy2(current_file, destination)

        print(f"Copied and renamed: {current_file} → {destination}")
    elif current_file.endswith('.ipynb'):
        print("Current file is a Jupyter notebook (.ipynb). Skipping copy.")
    else:
        print("Unsupported file type.")

if is_python_script():
    copy_and_rename_if_py(copy_filename)
else:
    print("This is a .ipynb file")
# -

k_eff_all,pkl0ezmocks,pkl2ezmocks,pkl4ezmocks,B000ezmocks,B202ezmocks = ExtractDataEZmock(tracer,z_str)
k_cov_all, mean_ezmocks_all, cov_array_all = covariance(k_eff_all,pkl0ezmocks,pkl2ezmocks,pkl4ezmocks,B000ezmocks,B202ezmocks, Nscaling = Vol)


# +
pole_selection=[isP0, isP2, isP4,isB000, isB202]
print(pole_selection)

kmin_pk=0.02; kmax_pk=k_max
kmin_bk=0.02; 
ranges=[[kmin_pk,kmax_pk],[kmin_pk,kmax_pk],[kmin_pk,kmax_pk],[kmin_bk,k_max_b0],[kmin_bk,k_max_b2]]

mask=pole_k_selection(k_cov_all,pole_selection,ranges)
#print(mask.shape)
#print(mask)
k_cov=k_cov_all[mask]

#k_cov.shape
k_points_pk = np.where((kmin_pk < k_data_2nd) & (k_data_2nd < kmax_pk)  & isP0)
k_points_b0 = np.where((kmin_bk < k_data_2nd) & (k_data_2nd < k_max_b0) & isB000)
k_points_b2 = np.where((kmin_bk < k_data_2nd) & (k_data_2nd < k_max_b2) & isB202)

data = np.concatenate((Pk_0_2nd[k_points_pk],Pk_2_2nd[k_points_pk],
                       B000_2nd[k_points_b0],B202_2nd[k_points_b2]))
kr_pk=k_data_2nd[k_points_pk]
kr_b0=k_data_2nd[k_points_b0]
kr_b2=k_data_2nd[k_points_b2]

numberofpk0points=len(Pk_0_2nd[k_points_pk])
numberofbk0points=len(B000_2nd[k_points_b0])
numberofbk2points=len(B202_2nd[k_points_b2])

cov_array=cov_array_all[np.ix_(mask, mask)]
totsim = 2000 #number of sims
n_data = len(data)
Hartlap = (totsim - 1.) / (totsim - n_data - 2.)
Hartlap
cov_arr = cov_array * Hartlap
cov_inv_arr = np.linalg.inv(cov_arr)

# +
N_ck=25  #kmaxThy = 0.01 * N_ck + 0.0025 + 0.0005
N_ck = max(int(k_max*100)+2,25)
print(N_ck)

k_thy_2nd_ext = np.linspace(0.0, 0.01*N_ck, 2 * N_ck * 5,endpoint=False) + 0.0025+0.0005 #+ 0.0025(move to first data bin) + 0.0005(move to center of bin)

ko_2nd_ext=k_data_2nd[0: 2 * N_ck]

m_bin_2nd_ext = np.zeros((len(ko_2nd_ext),len(k_thy_2nd_ext)))
m_bin_k_2nd_ext = np.zeros((len(ko_2nd_ext),len(k_thy_2nd_ext)))

for i,ki in enumerate(ko_2nd_ext):
    norm_2nd_ext = (1./3.)* ( (k_thy_2nd_ext[5*i + (5-1)])**3 - (k_thy_2nd_ext[5*i])**3 )
    for j in range(5):
        ff=((5-1)/5)
        m_bin_2nd_ext[i,5*i + j] = (k_thy_2nd_ext[5*i + j]**2)*0.001 / norm_2nd_ext * ff
        m_bin_k_2nd_ext[i,5*i + j] = k_thy_2nd_ext[5*i + j]


k_thy=k_thy_2nd_ext
m_bin=m_bin_2nd_ext

# ones=np.ones(5)
# def Ab2genbinning_ext(array,shift=0):
#     import matplotlib.pyplot as plt
#     for n in array:
#         print(n)
#         print(m_bin_2nd_ext[n,5*n : 5*(n+1)])
#         print(m_bin_k_2nd_ext[n,5*n : 5*(n+1)])
#         print(np.mean(m_bin_k_2nd_ext[n,5*n : 5*(n+1)]))
#         print(ko_2nd_ext[n])
#         print(k_thy_2nd_ext[n])
#         plt.ylim(0.8, 1.3)
#         plt.plot(m_bin_k_2nd_ext[n,5*n : 5*(n+1)],ones,'o')
#         plt.plot(ko_2nd_ext[n]-shift,1.2,'o')
#         plt.plot(k_thy_2nd_ext[ 5*n+2]-shift,1.22,'o')
#         plt.show()
#         print(' ')
#         #print(' ')
#     return None

# Ab2genbinning_ext([0,N_ck*2-1])


# +
kb_all=np.linspace(0.5*k_thy_2nd_ext[0], 0.28, 60)
k_ev_bk=np.vstack([kb_all,kb_all]).T

Mmatrices = FOLPS.Matrices()
Omfid = 0.31519186799


# -

def Folps(h, omega_cdm, omega_b, logA_s, b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir, a_vir_bk, k_ev):

    global sigma8, Omega_m
    
    "Fixed values: CosmoParams"
    #Omega_i = w_i/h² , w_i: omega_i
    #omega_b = 0.02237;             #Baryons
    #omega_cdm = 0.1200;            #CDM
    omega_ncdm = 0.00064420;        #massive neutrinos
    #h = 0.6736                     #H0/100
    A_s = np.exp(logA_s)/(10**10);  #A_s = 2.0830e-9;  
    n_s = 0.9649;
    z_pk = z_evaluation;                     #z evaluation
    
    CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
    
    "Fixed values: NuisanParams"
    #b1 = 1.0;      
    #b2 = 0.2;      
    #bs2 = -4/7*(b1 - 1);      
    b3nl = 32/315*(b1 - 1);
    alpha0 = 0.0;                    #only for reference - does not affect the final result
    alpha2 = 0.0;                 #only for reference - does not affect the final result 
    alpha4 = 0.0;                       
    ctilde = 0.0;    
    alphashot0 = 0.0;            #only for reference - does not affect the final result
    alphashot2 = 0.0;             #only for reference - does not affect the final result    
    PshotP = 1/0.0002118763;        ### it is completely degenerate with alphashot0
    
    NuisanParams = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                    ctilde, alphashot0, alphashot2, PshotP, a_vir]
    
    "linear cb power spectrum"
    ps = run_bacco(h = h, ombh2 = omega_b, omch2 = omega_cdm, omnuh2 = omega_ncdm,
                         As = A_s, ns = n_s, z = z_pk)
    
    sigma8 = ps['sigma8']
    Omega_m = ps['Omega_m']
    
    inputpkT = ps['k'], ps['pk']
        
    "Computing 1-loop corrections"
    LoopCorrections = FOLPS.NonLinear(inputpkl=inputpkT, CosmoParams=CosmoParams, EdSkernels=False)
    
    
    ##Pℓ,const
    Pkl0_const, Pkl2_const = FOLPS.RSDmultipoles_marginalized_const(k_ev, NuisanParams=NuisanParams, 
                                                                    Omfid = Omfid, AP=True)
    
    ##Pℓ,i=∂Pℓ/∂α_i
    Pkl0_i, Pkl2_i = FOLPS.RSDmultipoles_marginalized_derivatives(k_ev, NuisanParams=NuisanParams, 
                                                                  Omfid = Omfid, AP=True)
    
    #Binning for Pℓ,const
    Pkl0_const_mbin = m_bin @ Pkl0_const 
    Pkl2_const_mbin = m_bin @ Pkl2_const 
    
    Pl02_const_binning = np.concatenate((Pkl0_const_mbin[k_points_pk], Pkl2_const_mbin[k_points_pk]))  
    
    
    #Binning for Pℓ,i=∂Pℓ/∂α_i
    Pkl0_i_mbin = np.zeros((len(Pkl0_i), len(m_bin)))
    Pkl2_i_mbin = np.zeros((len(Pkl0_i), len(m_bin)))
    
    for ii in range(len(Pkl0_i)):
        Pkl0_i_mbin[ii, :] = m_bin @ Pkl0_i[ii]
        Pkl2_i_mbin[ii, :] = m_bin @ Pkl2_i[ii]

    #taking only k_points
    Pl0_i_binning = np.array([Pkl0_i_mbin[ii][k_points_pk] for ii in range(len(Pkl0_i))])
    Pl2_i_binning = np.array([Pkl2_i_mbin[ii][k_points_pk] for ii in range(len(Pkl2_i))])
    
    Pl02_i_binning_ = np.concatenate((Pl0_i_binning, Pl2_i_binning), axis = 1)

    Pl02_i_binning=np.zeros((len(Pkl0_i),len(data)))  #marginalizing over 5 parameters

    Pl02_i_binning[:,0:2*numberofpk0points]=Pl02_i_binning_



    if isB000 or isB202:
        
        pklir=FOLPS.pklIR_ini(LoopCorrections[0][0], LoopCorrections[0][1], LoopCorrections[1][1], h=h) 
        bisp_nuis_params = [b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir_bk]
        bisp_cosmo_params = [(omega_cdm+omega_b+omega_ncdm)/h**2,h]
        
        # precision=[5,7,7]
        B000__, B202__ = FOLPS.Bisp_Sugiyama(bisp_cosmo_params, bisp_nuis_params,
                                         pk_input=pklir, z_pk=z_pk, k1k2pairs=k_ev_bk,
                                            Omfid=Omfid)#, 
                                               #, Omfid=-1,precision=precision)
        B000_ = FOLPS.interp(k_thy,k_ev_bk[:,0],B000__)
        B202_ = FOLPS.interp(k_thy,k_ev_bk[:,0],B202__)
        B000_const_mbin = m_bin @ B000_ 
        B202_const_mbin = m_bin @ B202_
        Pl02_const_binning = np.concatenate((Pkl0_const_mbin[k_points_pk], Pkl2_const_mbin[k_points_pk],
                                             B000_const_mbin[k_points_b0], B202_const_mbin[k_points_b2])) 
        
        
    return ({'pl02_const':Pl02_const_binning, 'pl02_i':Pl02_i_binning})


#h, omega_cdm, omega_b, logA_s, b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir, a_vir_bk, k_ev
out=Folps(0.7, 0.12, 0.02, 3.0, 2, 0.1, -0.2, 0.2, 0.1,100,1000, 2,2, k_thy)
print('P0   points:', numberofpk0points)
print('B000 points:', numberofbk0points)
print('B202 points:', numberofbk2points)

# +
#priors range
h_min = 0.5; h_max = 0.9   
ocdm_min = 0.05; ocdm_max = 0.2                           
oncdm_min = 1e-5 ; oncdm_max = 0.0222   #M_\nu :(0.0009 - 2 eV)
#omega_b: Gaussian prior
ob_min = 0.0170684; ob_max = 0.0270684
logAs_min = 2.0; logAs_max = 4.0 

b1_min = 1e-5; b1_max = 10     #b1 cannot be 0: A(mu, f0)-> A(mu, f0/b1)
b2_min = -50; b2_max = 50
bs2_min = -50; bs2_max = 50
a_vir_min=0; a_vir_max=10; a_vir_EFT=0.0
a_vir_bk_min=0; a_vir_bk_max=15; a_vir_bk_EFT=0.0

c1_min = -2000; c1_max = 2000
c2_min = -2000; c2_max = 2000
Bshot_min = -50000; Bshot_max = 50000
Pshot_min = -50000; Pshot_max = 50000



def log_prior(theta):
    ''' The natural logarithm of the prior probability. '''

    lp = 0.
    
    # unpack the model parameters
    # h, omega_cdm, omega_b, logA_s, b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir, a_vir_bk = theta
    h, omega_cdm, omega_b, logA_s, b1, b2, bs2, a_vir = theta

    
    """set prior to 1 (log prior to 0) if in the range+
       and zero (-inf) outside the range"""
    
    #uniform (flat) priors
    if (h_min < h < h_max and
        ocdm_min < omega_cdm < ocdm_max and
        ob_min < omega_b < ob_max and
        #oncdm_min < omega_ncdm < oncdm_max and
        logAs_min < logA_s < logAs_max and
        b1_min  < b1  < b1_max and
        b2_min  < b2  < b2_max and
        bs2_min < bs2 < bs2_max and
        # c1_min  < c1  < c1_max and
        # c2_min  < c2  < c2_max and
        # Bshot_min < Bshot < Bshot_max and
        # Pshot_min <  Pshot <  Pshot_max and
        a_vir_min <= a_vir < a_vir_max #and
        # a_vir_bk_min <= a_vir_bk < a_vir_bk_max  
       ):
        lp = 0.
    
    else:
        lp = -np.inf   
     
    # Gaussian prior on omega_b
    omega_bmu = 0.02237         # mean of the Gaussian prior - QUIJOTE
    omega_bsigma = 0.00037        # standard deviation of the Gaussian prior - BBN (slack ShapeFit, Héctor)
    lp -= 0.5*((omega_b - omega_bmu)/omega_bsigma)**2

    Ppoisson=1/0.0002118763
    # c1v = 6.0/0.3**2; c2v = 0.0/0.3**2; 
    # Bshotv=Ppoisson *3/4.7 *0.1
    # Pshotv=Ppoisson *2/4.7 *0.8

    # bs2_mu = -4.0/7.0*(b1 - 1.0)
    # bs2_sigma = 4*4.0/7.0*(b1 - 1.0)
    # lp -= 0.5*((bs2 - bs2_mu)/bs2_sigma)**2

    if isB000 or isB202:
        
        Pshot_mu = 0.0      
        Pshot_sigma = Ppoisson*4.0     
        lp -= 0.5*((Pshot - Pshot_mu)/Pshot_sigma)**2
        
        Bshot_mu = 0.0     
        Bshot_sigma = Ppoisson*4.0    
        lp -= 0.5*((Bshot - Bshot_mu)/Bshot_sigma)**2
    
        c1_mu = 66.6       
        c1_sigma = 66.6*4.   
        lp -= 0.5*((c1 - c1_mu)/c1_sigma)**2
    
        c2_mu = 0.0     
        c2_sigma = 1.0*4.0  
        lp -= 0.5*((c2 - c2_mu)/c2_sigma)**2
   
    return lp


def log_likelihood(theta, Hexa = False):
    '''The natural logarithm of the likelihood.'''
    
    # unpack the model parameters
    # h, omega_cdm, omega_b, logA_s, b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir, a_vir_bk = theta
    h, omega_cdm, omega_b, logA_s, b1, b2, bs2, a_vir = theta

    default_derived = np.array([np.nan] * len(derived_params), dtype=np.float64)
    
    # condition to evaluate the model 
    if (h_min < h < h_max and
        ocdm_min < omega_cdm < ocdm_max and
        ob_min < omega_b < ob_max and
        #oncdm_min < omega_ncdm < oncdm_max and
        logAs_min < logA_s < logAs_max and
        b1_min  < b1  < b1_max and
        b2_min  < b2  < b2_max and
        bs2_min < bs2 < bs2_max and
        # c1_min  < c1  < c1_max and
        # c2_min  < c2  < c2_max and
        # Bshot_min < Bshot < Bshot_max and
        # Pshot_min <  Pshot <  Pshot_max and
        a_vir_min < a_vir < a_vir_max #and
        # a_vir_bk_min < a_vir_bk < a_vir_bk_max  
       ): 
        
        #evaluate the model
        
        c1,c2,Pshot,Bshot,a_vir_bk=0,0,0,0,0    
        md = Folps(h, omega_cdm, omega_b, logA_s, b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir, a_vir_bk, k_thy)
        
        md_const = md['pl02_const']; 
        
        if Hexa == False:
            #delete the array for alpha_4, no hexa (delete third row of model['pl02_i'])
            md_i = np.delete(md['pl02_i'], 2, 0);
        else:
            md_i = md['pl02_i'] 
        
        L0 = FOLPS.compute_L0(Pl_const=md_const, Pl_data=data, invCov=cov_inv_arr)
        L1i = FOLPS.compute_L1i(Pl_i=md_i, Pl_const=md_const, 
                                Pl_data=data, invCov=cov_inv_arr)
        L2ij = FOLPS.compute_L2ij(Pl_i=md_i, invCov=cov_inv_arr)
        
        #compute the inverse of L2ij
        invL2ij = np.linalg.inv(L2ij)
        
        #compute the determinat of L2ij
        detL2ij = np.linalg.det(L2ij)
        
        #marginalized likelihood
        term1 = FOLPS.startProduct(L1i, L1i, invL2ij)
        term2 = np.log(abs(detL2ij))
        
        L_marginalized = (L0 + 0.5 * term1 - 0.5 * term2) 

        derived = []
        if 'sigma8' in derived_params:
            derived.append(sigma8)
        if 'Omega_m' in derived_params:
            derived.append(Omega_m)
        derived = np.array(derived, dtype=np.float64)

        return L_marginalized, derived
        
    else:
        L_marginalized = 10e10
   
        return L_marginalized, default_derived

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, np.array([np.nan] * len(derived_params), dtype=np.float64)

    ll, derived = log_likelihood(theta)
    return lp + ll, derived


# -

start0 = np.array([0.6760,   #h
                   0.1186,   #ocdm
                   0.0218,   #ob
                   #0.0024,   #oncdm
                   3.0322,   #logAs
                   1.9608,   #b1
                   -0.8811,   #b2
                   -0.1,   #bs2
                   # 0.0,  #c1
                   # 0.0,  #c2
                   # 0.0,  #Pshot
                   # 0.0,  #Bshot
                   0.2,  #avir
                   # 0.6  #avir_bk
                  ])

# +
ndim = len(start0) # Number of parameters/dimensions (e.g. m and c)
nwalkers = 2 * ndim # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 50000 # Number of steps/iterations. (max number)

log_probability(start0)
# -


# In[ ]:


# +
start = np.array([start0 + 1e-3*np.random.rand(ndim) for i in range(nwalkers)])

 
backend = emcee.backends.HDFBackend(chains_filename)
##backend.reset(nwalkers, ndim)
###sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)


max_n = nsteps

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf



with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, 
                                    pool=pool)
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(start, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
            
            
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.005)
        if converged:
            break
        old_tau = tau



np.savetxt('autocorr_index'+str(index)+'.dat', np.transpose([autocorr[:index]]), 
           header = 'index ='+str(index)+',  mean_autocorr')


print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
)

