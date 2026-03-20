#!/usr/bin/env python
# coding: utf-8

# In[2]:

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

import matplotlib.pyplot as plt
import numpy as np
from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BAOCompressionObservable 
from desilike.observables import ObservableCovariance
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging
from desilike.parameter import Parameter, ParameterPrior


# In[2]:


# In[ ]:


######### Settings ######### 

#model: LCDM or HS
model = 'LCDM'

#Put 'True' to resume chain. 'False' to start from 0 steps
restart_chain = False
#Biasing and EFT parametrization: 'physical' or 'default' (non-physical)
prior_basis = 'physical' 

#max. wavenumber 
k_max = 0.201

#width for EFT and SN paramss
width_EFT = 12.5
width_SN0 = 2.0
width_SN2 = 5.0

#PT models: 'folps', 'pybird', 'lpt_velocileptors', 'rept_velocileptors'
pt_model = "EFT" 
# pt_model = "rept_velocileptors"

#samplers: 'cobaya', 'emcee'
sampler = 'cobaya'

# Set to 'True' to use emulators, or 'False' to disable them.
set_emulator = True

# R - 1 < GR_criteria 
GR_criteria = 0.05 

profiler= False

# In[3]:

# List of tracers
tracers = ['BGS_NGC', 'LRG1_NGC', 'LRG2_NGC', 'LRG3_NGC', 'ELG_NGC', 'QSO_NGC' ]  # Add more tracers as needed
tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO' ]  # Add more tracers as needed
chain_name = f'Chains/desiy1_fs-all_schoneberg2024-bbn_planck2018-ns-fixed_{prior_basis}-prior_{model}_'f'kmax{str(k_max).replace(".", "p")}' #fn to save the chain from the sampler

all_tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']
tracers_str = "all"
# if set(tracers) == all_tracers:
#     tracers_str = "all"
# else:
#     tracers_str = "-".join(tracers)

chain_name = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains/EFT/MCMC-fs_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_Ano_tight_eft_b3_free'

if profiler:
    chain_name = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains_profiler/base_EFT/Iminuit-fs_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_Ano_eft_18_ACPT'


# Define file paths for each tracer (in the same order as the tracers list above)
tracer_params = {
    0: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_default_FKP_lin_thetacut0.05.npy'
    },
    1: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.4-0.6_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.4-0.6_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_GCcomb_z0.4-0.6_default_FKP_lin_thetacut0.05.npy'
    },
    2: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.6-0.8_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.6-0.8_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_GCcomb_z0.6-0.8_default_FKP_lin_thetacut0.05.npy'
    },
    3: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.8-1.1_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.8-1.1_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_GCcomb_z0.8-1.1_default_FKP_lin_thetacut0.05.npy'
    },
    4: {
       'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_ELG_LOPnotqso_GCcomb_z1.1-1.6_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_ELG_LOPnotqso_GCcomb_z1.1-1.6_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_ELG_LOPnotqso_GCcomb_z1.1-1.6_default_FKP_lin_thetacut0.05.npy'
    },
    5: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_QSO_GCcomb_z0.8-2.1_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_QSO_GCcomb_z0.8-2.1_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_QSO_GCcomb_z0.8-2.1_default_FKP_lin_thetacut0.05.npy'
    }
}    



#Define a cosmology to get sigma_8, Omega_m and fR0
cosmo = Cosmoprimo(engine='class')
cosmo.init.params['H0'] = dict(derived=True)
cosmo.init.params['Omega_m'] = dict(derived=True)
cosmo.init.params['sigma8_m'] = dict(derived=True) 
# cosmo.init.params['fR0'] = dict(derived=False, latex ='f_{R0}')
fiducial = DESI() #fiducial cosmology

#Update cosmo priors
for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA', 'tau_reio']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            # cosmo.params[param].update(fixed = True)
            # cosmo.params[param].update(value = 0.9649)
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042})
    if param == 'omega_b':
            cosmo.params[param].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.5,0.9]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.05, 0.2]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})
    if param == 'fR0':
        if model != 'HS':
            print('model LCDM')
            cosmo.params[param].update(fixed=True, value=0.0)
        else:
            print('model HS')
            cosmo.params[param].update(
            prior={'dist': 'uniform', 'limits': [0, 5e-5]},
            fixed=False,
            ref={'limits': [0, 5e-5]}
            )

#Define tracer types and their corresponding redshifts
tracer_redshifts = {
    'LRG1': 0.510, 'LRG1_SGC': 0.510,
    'LRG2': 0.706, 'LRG2_SGC': 0.706,
    'LRG3': 0.930, 'LRG3_SGC': 0.930,
    'QSO': 1.491, 'QSO_SGC': 1.491,
    'ELG': 1.317, 'ELG_SGC': 1.317,
    'BGS': 0.295, 'BGS_SGC': 0.295
}

#Initialize an empty list to store the theory objects
theories = []
all_tracer_sigma8_fid = {
                        'BGS': 0.69376997,
                        'LRG1': 0.6208944,
                        'LRG2': 0.56378872,
                        'LRG3': 0.50823223,
                        'ELG': 0.43197292,
                        'QSO': 0.4041147
                        }
#Iterate over each tracer and create the corresponding theory object
for tracer in tracers:
    if tracer in tracer_redshifts:
        z = tracer_redshifts[tracer]
        sigma8_fid = all_tracer_sigma8_fid[tracer]
    else:
        print(f'Invalid tracer: {tracer}. Skipping.') 
        continue

    #Create the template and theory objects
    template = DirectPowerSpectrumTemplate(fiducial = fiducial,cosmo = cosmo, z=z) #cosmology and fiducial cosmology defined above
    theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, prior_basis=prior_basis,A_full=False, b3_coev=False)

    #Update bias and EFT priors
    #theory.params['bs'].update(fixed=True)
    if prior_basis == 'physical':
        theory.params['b1p'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
        theory.params['b2p'].update(prior = {'dist':'uniform','limits': [-50, 50]})
        theory.params['bsp'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 40})
        theory.params['b3p'].update(prior = {'dist': 'norm', 'loc': 32/315, 'scale': 2*32/315})
        theory.params['b3p'].update(fixed=False)
        theory.params['alpha0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha4p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['sn0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN0})
        theory.params['sn2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN2})
        # theory.params['sn4p'].update(fixed=True)
        if pt_model=='EFT':
            theory.params['X_FoG_pp'].update(fixed=True)
        elif pt_model=='rept_velocileptors':
            None
        else:
            theory.params['X_FoG_pp'].update(prior = {'dist':'uniform','limits': [0, 10]})
    else:
        theory.params['b1'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
        theory.params['b2'].update(prior = {'dist':'uniform','limits': [-50, 50]})
        theory.params['bs'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 20})
        theory.params['alpha0'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha2'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha4'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['sn0'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN0})
        theory.params['sn2'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN2})
        theory.params['sn4'].update(fixed=True)
        # if pt_model=='EFT':
        #     theory.params['X_FoG_p'].update(fixed=True)
        # else:
        #     theory.params['X_FoG_p'].update(prior = {'dist':'uniform','limits': [0, 10]})
    
    #Append the theory object to the list
    theories.append(theory)

#Print used priors
for param in theory.all_params:
    print(param,':',theory.all_params[param].prior)

#Define a function to create an observable
def create_observable(data_fn, wmatrix_fn, covariance_fn, theory, tracer_index):
    #Load and process covariance
    covariance = ObservableCovariance.load(covariance_fn)
    covariance = covariance.select(xlim=(0.02, k_max), projs=[0, 2])
    
    #Create and return the observable
    return TracerPowerSpectrumMultipolesObservable(
        data=data_fn,
        covariance=covariance,
        klim={ell: [0.02, k_max, 0.005] for ell in [0, 2]},
        theory=theories[tracer_index],
        wmatrix=wmatrix_fn,
        kin=np.arange(0.001, 0.35, 0.001),
    )

#Create observables for each tracer
observables = [create_observable(params['data_fn'], params['wmatrix_fn'], params['covariance_fn'],theories, i) 
                for i, params in tracer_params.items()]


#Create an emulated theory for each tracer
if set_emulator:
    #Create an emulated theory for each tracer
    for i in range(len(theories)):
        emulator_filename = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Emulator_y1/emu_-{tracers_str}_{str(all_tracers[i])}.npy'
        
        if os.path.exists(emulator_filename):
            print(f"FS emulator {i} already exists, loading it now")
            emulator = EmulatedCalculator.load(emulator_filename)  
            theories[i].init.update(pt=emulator)
        else:
            print(f" Computing FS emulator {i}")
            theories[i] = observables[i].wmatrix.theory
            emulator = Emulator(theories[i].pt, engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulator.save(emulator_filename)
            theories[i].init.update(pt=emulator.to_calculator())
print('FS theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED for FS; proceeding without emulation')



#Analytic marginalization over eft and nuisance parameters
for i in range(len(theories)): 
    if prior_basis == 'physical':
        params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
        params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for param in params_list:    
        theories[i].params[param].update(derived = '.marg')

        
#Rename the eft and nuisance parameters to get a parameter for each tracer (i.e. QSO_alpha0, QSO_alpha2, BGS_alpha0,...)        
for i in range(len(theories)):    
    for param in theories[i].init.params:
        # Update latex just to have better labels
        param.update(namespace='{}'.format(tracers[i])) 
        

#Create a likelihood per theory object
setup_logging()
Likelihoods = []
for i in range(len(theories)):
    Likelihoods.append(ObservablesGaussianLikelihood(observables = [observables[i]]))


#Sum the likelihoods and initialize
likelihood = SumLikelihood(likelihoods = (Likelihoods))
print("likelihood: ",likelihood())
print("Test Successful!")
# for i in range(len(theories)):
#     observables[i].plot(fn=f'plot{i}.png')




# In[ ]:


#Run the sampler and save the chain
from desilike.samplers import EmceeSampler, MCMCSampler

if profiler:
    from desilike.profilers import MinuitProfiler
    profiler = MinuitProfiler(likelihood, seed=42,save_fn=chain_name)
    profiles = profiler.maximize(niterations=5,)
else: 
    if sampler == 'cobaya':
        if restart_chain is False:
            sampler = MCMCSampler(likelihood, save_fn = chain_name)
            sampler.run(check={'max_eigen_gr': 0.03})
        else:
            sampler = MCMCSampler(likelihood ,save_fn = chain_name, 
                                  chains=f'{chain_name}.npy')
            sampler.run(check={'max_eigen_gr': 0.03})
        
    else:
        if restart_chain is False:
            sampler = EmceeSampler(likelihood ,save_fn = chain_name)
            sampler.run(check={'max_eigen_gr': 0.03})
        else:
            sampler = EmceeSampler(likelihood ,save_fn = chain_name, 
                                   chains=f'{chain_name}.npy')
            sampler.run(check={'max_eigen_gr': 0.03})


# In[ ]:




