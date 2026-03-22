import sys

sys.path.insert(0, "/global/u1/p/prakharb/desilike-folps-jax-dev")
#sys.path.insert(0, "/global/u1/p/prakharb/desilike")
#sys.path.insert(0, "/global/u1/p/prakharb/cosmoprimo")
import desilike, inspect
print(inspect.getfile(desilike))
import os
os.environ["FOLPS_BACKEND"] = "jax" 
import folps, inspect
print(inspect.getfile(folps))
import desilike, inspect
print(inspect.getfile(desilike))
from mike_data_tools import *            #Change this depending on where mike data tools is stored


import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles,  DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles, FOLPSAXTracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles, FOLPSTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable 
from desilike.observables import ObservableCovariance
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging
from desilike.parameter import Parameter, ParameterPrior
import argparse
import sys, os, shutil
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
from datetime import datetime







# In[ ]:


######### Settings #########

#model: LCDM or HS
model = 'LCDM'


base_dir = '/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains'
#Put 'True' to resume chain. 'False' to start from 0 steps
restart_chain = False
#Biasing and EFT parametrization: 'physical' or 'standard'
prior_basis = 'standard' #Prior to be used 

damping= 'lor'  #Choose from 'lor', 'exp', 'vdg'

kr_max = 0.201
kr_b0_max = 0.15
kr_b2_max = 0.12

hexa = False
bispectrum= False

if bispectrum==False:
    kr_b0_max = None
    kr_b2_max = None

#No need to change these
width_EFT = 125
width_SN0 = 20
width_SN2 = 50

pt_model = "EFT"  #Choose b/w folpsD and EFT, TNS

sampler = 'cobaya'

set_emulator = True

A_full_status= True

b3_coev = True

GR_criteria = 0.01  # R - 1 < GR_criteria 





# List of tracers
# tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']  # Add more tracers as needed
tracers = ['LRG2']
#tracers = args.tracers
all_tracers = {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}

if set(tracers) == all_tracers:
    tracers_str = "all"
else:
    tracers_str = "_".join(tracers)



chain_name = (
    f"{base_dir}/cxxx_mocks_chains/c004/base/{tracers_str}"
    f"_{'std' if prior_basis == 'standard' else 'phys'}"
    f"_kr{kr_max:.3f}"
    f"{f'_kb0{kr_b0_max:.3f}_kb2{kr_b2_max:.3f}' if bispectrum else ''}"
    f"{'_hexa' if hexa else ''}"
    f"_{pt_model}"
    f"_{'Afull' if A_full_status else 'Ano'}"
    f"_{'b3_coev' if b3_coev else 'b3_samp'}"
    f"{f'_damping_{damping}' if damping != 'lor' else ''}"
)
# chain_name= f'chains/test_mocks_std_folps_scheme_newcoev_alejandro_classpt_eft'

# chain_name = f'{comp}_{tracer}_{z}_{kr_max}_Afull_{A_full_status}'
# chain_name = f'chains/MCMC-fs_abacus2gen_new_folpsv2_fk_ps_0.351_Afull_false'
# if not set_emulator:
#     chain_name += '_noemu'
print(chain_name)



######## 

# No need to change anything beyond this

#########

from desilike import ParameterCollection

def make_params(prior_basis, width_EFT, width_SN0, width_SN2,pt_model='folpsD',b3_coev=True):
    params = ParameterCollection()

    if prior_basis == 'physical':
        # Shared params
        # params['b1p']= {'prior': {'dist': 'uniform', 'limits':[1e-5,3]}}
        # params['b2p']= {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        # params['bsp']= {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['b1p']= {'prior': {'dist': 'uniform', 'limits':[1e-5,10]}}
        params['b2'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        if b3_coev:
            params['b3p'] = {'fixed':True}
        else:
            params['b3p']= {'prior': {'dist': 'norm', 'loc': 23/42, 'scale': 1},'fixed':False}
        
        # PS-only
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': width_EFT, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        if pt_model=='EFT': 
            params['X_FoG_pp'] = {'fixed':True}
        else: 
            params['X_FoG_pp'] = {'prior': {'dist':'uniform','limits': [0, 10]}} 
        # BS-only → if physical, no c1,c2,Pshot,Bshot,X_FoG_b
        # params['c1'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        # params['c2'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        # params['Pshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        # params['Bshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        # params['X_FoG_b'] = {'prior': {'dist':'uniform','limits': [0, 15]}}

    else:
        # Shared params
        params['b1'] = {'prior': {'dist':'uniform','limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed':True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 25}}#TBD
        # params['b3'] = {'fixed':True}

        # PS-only
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        # params['alpha0'] = {'value':20,'fixed':True}
        # params['alpha2'] = {'value':-58.8,'fixed':True}
        # params['alpha4'] = {'value':0.0,'fixed':True}
        # params['sn0'] = {'value':-0.073,'fixed':True}
        # params['sn2'] = {'value':-6.38,'fixed':True}
        if pt_model=='EFT':
            params['X_FoG_p'] = {'fixed':True}
            params['X_FoG_b'] = {'fixed':True}
        else: 
            params['X_FoG_p'] = {'prior': {'dist':'uniform','limits': [0, 10]}} 
            params['X_FoG_b'] = {'prior': {'dist':'uniform','limits': [0, 15]}}
        # params['X_FoG_p'] = {'fixed':True}  # fixed in your snippet

        # BS-only
        params['c1'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        params['c2'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        params['Pshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        params['Bshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        

    return params




#Define a cosmology to get sigma_8, Omega_m and fR0
cosmo = Cosmoprimo(engine='class')
cosmo.init.params['H0'] = dict(derived=True)
cosmo.init.params['Omega_m'] = dict(derived=True)
cosmo.init.params['sigma8_m'] = dict(derived=True) 
#cosmo.init.params['fR0'] = dict(derived=False, latex ='f_{R0}')
fiducial = DESI() #fiducial cosmology

#Update cosmo priors
for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA', 'tau_reio']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            cosmo.params[param].update(fixed = True)
            # cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042})
    if param == 'omega_b':
            # cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': 0.00055})
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})  #From simulations
        # ,ref={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00015}
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.5,0.9]})
        # ,ref={'dist': 'norm', 'loc': 0.6736, 'scale': 0.005}
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.05, 0.2]})
        # ,ref={'dist': 'norm', 'loc': 0.12, 'scale': 0.0012}
    #if param == 'm_ncdm':
        #cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.0, 5]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})
        # ,ref={'dist': 'norm', 'loc': 3.036394, 'scale': 0.014}


#Define tracer types and their corresponding redshifts
all_tracer_params = {
    'BGS': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.npy'
    },
    'LRG1': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.4-0.6.npy'
    },
    'LRG2': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.6-0.8.npy'
    },
    'LRG3': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.8-1.1.npy'
    },
    'ELG': {
       'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_ELG_LOPnotqso_GCcomb_z1.1-1.6.npy'
    },
    'QSO': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_QSO_GCcomb_z0.8-2.1.npy'
    }
}  

all_tracer_redshifts = {
                        'BGS': 0.295,
                        'LRG1': 0.5,
                        'LRG2': 0.725,
                        'LRG3': 0.930,
                        'ELG': 1.317,
                        'QSO': 1.4
                        }

all_tracer_sigma8 = {
                        'BGS': 0.69376997,
                        'LRG1':0.62404056,
                        'LRG2': 0.53930063,
                        'LRG3': 0.50823223,
                        'ELG': 0.43197292,
                        'QSO': 0.41825647
}

tracer_params = {index: all_tracer_params[tracer] for index, tracer in enumerate(tracers)}
tracer_redshifts = {tracer: all_tracer_redshifts[tracer] for tracer in tracers}



#Iterate over each tracer and create the corresponding theory object
# Theories container: dict of dicts
theories = {}

for tracer in tracers:
    z = tracer_redshifts[tracer]
    sigma8_fid= all_tracer_sigma8[tracer]
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)

    # PS theory selection
    if pt_model == "rept_velocileptors":
        ps_theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template,
                                                                   prior_basis=prior_basis)
    else:
        if pt_model == 'TNS':
            ps_theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template,
                                                         prior_basis=prior_basis, A_full = A_full_status,b3_coev=b3_coev,damping=damping,remove_DeltaP=True)
        else:
               ps_theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template,
                                                         prior_basis=prior_basis, A_full = A_full_status,b3_coev=b3_coev,damping=damping) 
        # ps_theory = FOLPSTracerPowerSpectrumMultipoles(template=template,
        #                                                  prior_basis=prior_basis)

    # BS theory always FOLPSv2 in your snippet
    bs_theory = FOLPSv2TracerBispectrumMultipoles(template=template,
                                                  prior_basis=prior_basis)

    # Store
    theories[tracer] = {"ps": ps_theory, "bs": bs_theory}

    # --- Update parameters ---
    params = make_params(prior_basis, width_EFT, width_SN0, width_SN2, pt_model=pt_model,b3_coev=b3_coev)
    for name, p in params.items():
        for comp in ["ps", "bs"]:
            if name in theories[tracer][comp].params:
                theories[tracer][comp].params[name].update(p)


    # print("X_FoG_p=",params['X_FoG_pp'].value)
    # print("b3=",params['b3p'].value)


        
    

    # for param in ps_theory.all_params:
    #     print(param,':',ps_theory.all_params[param].prior)
    # for param in bs_theory.all_params:
    #     print(param,':',bs_theory.all_params[param].prior)

#Define data vector and covariance using mike data tools

def load_data_vector_and_covariance(tracer='LRG2',z_ev=0.8,k_max=0.301,k_max_b0=None,k_max_b2=None,P4=False):
    k_min=0.02
    # k_max=0.351
    # k_max_b0 = 0.15
    # k_max_b2 = 0.12

    isP0, isP2, isP4 =True, True, False
    isB000, isB202 = True, True
    if P4:
        isP4 = True
    if k_max_b0==None and k_max_b2==None: 
        isB000, isB202 = False, False
        k_max_b0 = 0.08
        k_max_b2 = 0.08   #Assign some placeholder values
    

    Vol=1

    if (tracer=='LRG1' or tracer=='LRG2'):
        tracer= 'LRG'
    else: 
        tracer=tracer
    z_evaluation=z_ev
    z_cov = 0.8
    z_str=f"z{z_cov:.3f}"
    
    

    # print(chains_filename)
        

    # +
    # k_data_2nd,pkl0_2nd_all,pkl2_2nd_all,pkl4_2nd_all,B000_2nd_all,B202_2nd_all = ExtractDataAbacusSummit(tracer,z_str,                                             subtract_shot=True)
    # # k_data_2nd,pkl0_2nd_all,pkl2_2nd_all,pkl4_2nd_all,B000_2nd_all,B202_2nd_all =  ExtractDataAbacusSummit_additionalcosmologies('c000')
    # print(pkl0_2nd_all.shape)
    # Pk_0_2nd = np.mean(pkl0_2nd_all,axis = 0)
    # Pk_2_2nd = np.mean(pkl2_2nd_all,axis = 0)
    # Pk_4_2nd = np.mean(pkl4_2nd_all,axis = 0)
    # B000_2nd = np.mean(B000_2nd_all,axis = 0)
    # B202_2nd = np.mean(B202_2nd_all,axis = 0)
    # k_data_2nd,Pk_0_2nd,Pk_2_2nd,Pk_4_2nd,B000_2nd,B202_2nd =  ExtractDataAbacusSummit_additionalcosmologies('c000')
    # k_data_2nd,Pk_0_2nd,Pk_2_2nd,Pk_4_2nd,B000_2nd,B202_2nd =  ExtractDataAbacusSummit_additionalcosmologies('c000')
    # Read covariance
    k_data_2nd,Pk_0_2nd,Pk_2_2nd,Pk_4_2nd,B000_2nd,B202_2nd =  ExtractDataAbacusSummit_additionalcosmologies('c004')
    k_eff_all,pkl0ezmocks,pkl2ezmocks,pkl4ezmocks,B000ezmocks,B202ezmocks = ExtractDataEZmock(tracer,z_str)
    k_cov_all, mean_ezmocks_all, cov_array_all = covariance(k_eff_all,pkl0ezmocks,pkl2ezmocks,pkl4ezmocks,B000ezmocks,B202ezmocks, Nscaling = Vol)



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

    data = np.concatenate((Pk_0_2nd[k_points_pk],Pk_2_2nd[k_points_pk],Pk_4_2nd[k_points_pk],
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
    # cov_inv_arr = np.linalg.inv(cov_arr)
    
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

    kb_all=np.linspace(0.5*k_thy_2nd_ext[0], 0.28, 60)
    k_ev_bk=np.vstack([kb_all,kb_all]).T
    
    
    # np.save("./desilike_mcmc/kr_b0.npy",kr_b0)
    # np.save("./desilike_mcmc/data_b0.npy",data)
    # np.save("./desilike_mcmc/cov_b0.npy",cov_array)
    return data, cov_array, kr_pk,kr_b0, kr_b2, k_thy, kb_all,m_bin,k_points_pk,k_points_b0,k_points_b2

def load_data_vector_and_covariance_files():
    kr_pk = np.load("./desilike_mcmc/kr_pk.npy")
    data = np.load("./desilike_mcmc/data.npy")
    cov_arr = np.load("./desilike_mcmc/cov.npy")
    return data, cov_arr, kr_pk





#Define a function to create an observable
def create_observable(comp,tracer='LRG2',z_ev=0.8,k_max=0.301,k_max_b0=None,k_max_b2=None,P4=False):
    global cov_arr
    data, cov_arr, kr_pk, kr_b0, kr_b2, k_thy, kb_all, m_bin, k_points_pk, k_points_b0, k_points_b2 = load_data_vector_and_covariance(tracer,z_ev,k_max,k_max_b0,k_max_b2,P4)
    print(kr_pk.shape)
    # cov_arr *= 0.2

    if comp == "ps":
        from scipy.linalg import block_diag

        # build window matrix for ell=0 and ell=2 separately
        wmatrix_pk = m_bin[np.asarray(k_points_pk).ravel(), :]   # (57, 320)
        
        # each ell gets its own mapping (57, 320)
        wmatrix_ell0 = wmatrix_pk
        wmatrix_ell2 = wmatrix_pk
        
        # block diagonal → (114, 640)
        wmatrix = block_diag(wmatrix_ell0, wmatrix_ell2)
        return TracerPowerSpectrumMultipolesObservable(
            data=data[:2*len(kr_pk)],
            covariance=cov_arr[:2*len(kr_pk), :2*len(kr_pk)],
            theory=theories[tracer]["ps"],
             kin = k_thy,
             ells=[0, 2], k=kr_pk, wmatrix=wmatrix
        )
      
    elif comp == "bs":
        start = len(data) - (len(kr_b0) + len(kr_b2))
    
        # Select rows of m_bin corresponding to b0 and b2 observed k-points
        wmatrix_b0 = m_bin[np.asarray(k_points_b0).ravel(), :]  # shape (len(kr_b0), n_theory)
        wmatrix_b2 = m_bin[np.asarray(k_points_b2).ravel(), :]  # shape (len(kr_b2), n_theory)
    
        # Block diagonalize to keep ell=0, ell=2 separate
        from scipy.linalg import block_diag
        wmatrix = block_diag(wmatrix_b0, wmatrix_b2)  # shape (len(kr_b0)+len(kr_b2), 2*n_theory)
    
        return TracerPowerSpectrumMultipolesObservable(
            data=data[start:],
            covariance=cov_arr[start:, start:],
            theory=theories[tracer]["bs"],
            kin=k_thy,   # provide theory grid for both multipoles
            ells=[0, 2],
            k=[kr_b0,kr_b2],  # observed bin centers
            wmatrix=wmatrix
        )

#Create observables for each tracer
observables = {}
for tracer in tracers:
    observables[tracer] = {
        "ps": create_observable( "ps",tracer,tracer_redshifts[tracer],kr_max,kr_b0_max,kr_b2_max,hexa)
    }



if set_emulator:
    for tracer in tracers:
        for comp in ["ps"]:  # handle PS and BS separately
            obs = observables[tracer][comp]

            # emulator_filename = 'Emulator_test_sims_ps/ps_emu_LRG2_LRG2_0.201_folpsax.npy'
            
            emulator_filename = f'{base_dir}/Emulators_c004/Emulator_{comp}/{comp}_emu_{tracer}_z{z}_{kr_max}_Afull_{A_full_status}.npy'
            os.makedirs(os.path.dirname(emulator_filename), exist_ok=True)

            if os.path.exists(emulator_filename):
                print(f"{comp.upper()} emulator for tracer {tracer} already exists, loading it now")
                emulator = EmulatedCalculator.load(emulator_filename)
                theories[tracer][comp].init.update(pt=emulator)
                # obs.theory.init.update(pt=emulator)

            else:
                print(f"Fitting {comp.upper()} emulator for tracer {tracer}")
                # Start from the underlying PT theory
                theory = obs.wmatrix.theory

                emulator = Emulator(
                    theory.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=4)
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                emulated_pt.save(emulator_filename)
                theories[tracer][comp].init.update(pt=emulated_pt)
                # obs.theory.init.update(pt=emulated_pt)

            

print('All theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED, proceeding without emulation')



#Analytic marginalization over eft and nuisance parameters
for i in (tracers): 
    if prior_basis == 'physical':
        params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
         params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for param in params_list:    
        theories[i]['ps'].params[param].update(derived = '.marg')
        
    # theories[i]['ps'].params['b3'].update(derived = '32/315*({b1}-1)')
    # print("b1=",theories[i]['ps'].params['b1'].value,"b3=",theories[i]['ps'].params['b3'].value)
    
        
#Rename the eft and nuisance parameters to get a parameter for each tracer (i.e. QSO_alpha0, QSO_alpha2, BGS_alpha0,...)        
# for i in range(len(theories)):    
    for param in theories[i]['ps'].init.params:
        # Update latex just to have better labels
        param.update(namespace='{}'.format(i)) 
    for param in theories[i]['ps'].all_params:
        print(param,':',theories[i]['ps'].all_params[param].prior)
        

#Create a likelihood per theory object
setup_logging()
Likelihoods = []
for tracer in tracers:
        Likelihoods.append(ObservablesGaussianLikelihood(
        observables[tracer]['ps']
        ))
   # observables=[observables[tracer]['ps'],observables[tracer]['bs']],
   #          covariance=cov_arr
likelihood = SumLikelihood(Likelihoods)








import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--run_chains", action="store_true", help="To run a chain")
parser.add_argument("--plot_bestfit", action="store_true", help="Plot Bestfit")
parser.add_argument("--plot_chains", action="store_true", help="Plot Chains")
parser.add_argument("--test", action="store_true", help="testing")    

args = parser.parse_args()




from desilike.samples import Chain
import matplotlib.pyplot as plt
from pathlib import Path
from desilike import setup_logging
setup_logging()

def load_chain(fi, burnin=0.3):
    from desilike.samples import Chain
    # chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
    chains = [Chain.load(fi).remove_burnin(burnin)]
    chain = chains[0].concatenate(chains)
    print(f'chain: {chain}')
    return chain


if args.test:
    print("likelihood: ",likelihood())
    print(likelihood.varied_params)
    print("Test Successful!")
    
    

if args.plot_bestfit:
    chain_path = Path(f'{chain_name}.npy')
    chain = load_chain(chain_path)
    likelihood(**chain.choice(index='argmax', input=True))
    observables['LRG2']['ps'].plot(fn='test_bestfit.png',kw_save={'dpi':250})




if args.plot_chains:
    chain_path = Path(f'{chain_name}.npy')
    chain = load_chain(chain_path)
    samples2 = chain.to_getdist()

    import h5py

    filename = "/global/cfs/cdirs/desicollab/users/isaacmgm/Abacus_2ndGen_Fits/folpsDBaccoemu/chains/c_FolpsEFT_LRG_z0.800_Pkmax-0.201_bsfree.h5"
    with h5py.File(filename, "r") as f:
        print(list(f.keys()))  # Shows top-level datasets/groups
        print(list(f['mcmc'].keys()))
    import emcee
    import numpy as np
    
    backend = emcee.backends.HDFBackend(filename, read_only=True)
    
    # Get total chain shape: (nwalkers, nsteps, ndim)
    chain_shape = backend.get_chain().shape
    nwalkers, nsteps, ndim = chain_shape
    
    # Set burn-in as 30% of total steps
    burnin = int(0.5 * nsteps)
    
    # Get flat chain, discarding burn-in
    samples = backend.get_chain(discard=burnin, flat=True)  # shape: (n_samples, ndim)
    
    # Select first 4 params and downsample by 10 for plotting speed
    samples_subset = samples[:, :4][::10]

    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    

    import matplotlib as mpl
    
    planck_truths = {
        'h': 0.6736,
        'omega_cdm': 0.12,
        'omega_b': 0.02237,
        'logA': np.log(10**10 * 2.0830e-9)  # log(10^10 A_s), or set to your matching definition
    }
    # High-resolution Retina output (for notebooks)
    
    
    # Use LaTeX for all matplotlib text rendering
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 18
    
    param_names = ['h', 'omega_cdm', 'omega_b', 'logA']
    param_labels = [r'$h$', r'$\Omega_{\rm cdm}$', r'$\Omega_{\rm b}$', r'$\log(10^{10}A_s)$']
    
    gdsamples = MCSamples(samples=samples_subset, names=param_names, labels=param_labels)
    
    g = plots.get_subplot_plotter()
    
    # Increase axis font sizes
    g.settings.axes_fontsize = 20
    g.settings.lab_fontsize = 20
    
    g.settings.line_labels = False
    
    legend_labels = ['Chain 1', 'Chain 2']
    
    g.triangle_plot(
        [gdsamples,samples2],
        params=param_names,
        filled=[False, True],  # First is dotted, second is filled
        line_args=[{'ls': '--', 'color': 'black'}, {'lw': 1.2, 'color': 'steelblue'}],
        contour_colors=['black', 'steelblue'],
        contour_ls=['--', '-'], 
        markers = planck_truths# Make contours dotted for both chains
    )
    g.add_legend(legend_labels=[r"FolpsEFT_LRG_z0.800_Pkmax-0.201_bsfree","desilike FOLPSv2 kmax=0.201 (A_full = False)"],bbox_to_anchor=(0.5, 3.95),fontsize=18)
    g.export("test_chains.png")

if args.run_chains: 
    def load_chain(fi, burnin=0.3):
        from desilike.samples import Chain
        # chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
        chains = [Chain.load(fi).remove_burnin(burnin)]
        chain = chains[0].concatenate(chains)
        print(f'chain: {chain}')
        return chain
    
    cov_chain_path = Path('/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Abacus2gen_chains/base/LRG2_std_kr0.301_folpsD_Afull_b3_coev.npy')
    cov_chain = load_chain(cov_chain_path,burnin=0.3)
    # cov_chain_path = Path('/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Abacus2gen_chains/base/test_cov.npy')
    # cov_chain = load_chain(cov_chain_path,burnin=0.3)
    

#Run the sampler and save the chain
    from desilike.samplers import EmceeSampler, MCMCSampler
    
    if sampler == 'cobaya':
        if restart_chain is False:
            sampler = MCMCSampler(likelihood, save_fn = chain_name,covariance=cov_chain)
            
            sampler.run(check={'max_eigen_gr': GR_criteria})
        else:
            sampler = MCMCSampler(likelihood ,save_fn = chain_name, 
                                  chains=f'{chain_name}.npy')
            #print(sampler.diagnostics)     # includes R-1, acceptance rate, etc.
            #print(sampler.converged)       # 
            sampler.run(check={'max_eigen_gr': GR_criteria})
        
    else:
        if restart_chain is False:
            sampler = EmceeSampler(likelihood ,save_fn = chain_name)
            sampler.run(check={'max_eigen_gr': GR_criteria})
        else:
            sampler = EmceeSampler(likelihood ,save_fn = chain_name, 
                                   chains=f'{chain_name}.npy')
            sampler.run(check={'max_eigen_gr': GR_criteria})