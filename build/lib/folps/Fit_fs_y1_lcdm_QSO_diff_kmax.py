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


######### Settings ######### 

#model: LCDM or HS
model = 'LCDM'

#Put 'True' to resume chain. 'False' to start from 0 steps
restart_chain = False
#Biasing and EFT parametrization: 'physical' or 'default' (non-physical)
prior_basis = 'physical' 

#max. wavenumber 
k_max = 0.301

#width for EFT and SN paramss
width_EFT = 12.5
width_SN0 = 2.0
width_SN2 = 5.0

#PT models: 'folps', 'pybird', 'lpt_velocileptors', 'rept_velocileptors'
pt_model = "folpsD" 

#samplers: 'cobaya', 'emcee'
sampler = 'cobaya'

# Set to 'True' to use emulators, or 'False' to disable them.
set_emulator = True

# R - 1 < GR_criteria 
GR_criteria = 0.05 


# In[3]:


# List of tracers
tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']  # Add more tracers as needed

k_max_array = [0.301,0.301,0.301,0.301,0.301,0.341]
all_tracers = {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}
all_tracer_kmax = {
                        'BGS': 0.301,
                        'LRG1': 0.301,
                        'LRG2': 0.301,
                        'LRG3': 0.301,
                        'ELG': 0.301,
                        'QSO': 0.341
                        }

all_tracer_indices = {
                        'BGS': 0,
                        'LRG1': 1,
                        'LRG2': 2,
                        'LRG3': 3,
                        'ELG': 4,
                        'QSO': 5
                        }
if set(tracers) == all_tracers:
    tracers_str = "all"
else:
    tracers_str = "-".join(tracers)

chain_name = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains/base_QSO_0.341/MCMC-fs_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_Ano_tight_eft'
# chain_name = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains/base/MCMC-fs+bao_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_nsfree_{prior_basis}-prior_{model}_{pt_model}_Ano_tight_eft'
if not set_emulator:
    chain_name += '_noemu'
#print(chain_name)


# Define file paths for each tracer (in the same order as the tracers list above)


all_tracer_params = {
    'BGS': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_default_FKP_lin_thetacut0.05.npy'
    },
    'LRG1': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.4-0.6_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.4-0.6_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_GCcomb_z0.4-0.6_default_FKP_lin_thetacut0.05.npy'
    },
    'LRG2': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.6-0.8_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.6-0.8_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_GCcomb_z0.6-0.8_default_FKP_lin_thetacut0.05.npy'
    },
    'LRG3': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.8-1.1_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.8-1.1_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_GCcomb_z0.8-1.1_default_FKP_lin_thetacut0.05.npy'
    },
    'ELG': {
       'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_ELG_LOPnotqso_GCcomb_z1.1-1.6_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_ELG_LOPnotqso_GCcomb_z1.1-1.6_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_ELG_LOPnotqso_GCcomb_z1.1-1.6_default_FKP_lin_thetacut0.05.npy'
    },
    'QSO': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_QSO_GCcomb_z0.8-2.1_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_QSO_GCcomb_z0.8-2.1_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_QSO_GCcomb_z0.8-2.1_default_FKP_lin_thetacut0.05.npy'
    }
}    


#Define tracer types and their corresponding redshifts
all_tracer_redshifts = {
                        'BGS': 0.295,
                        'LRG1': 0.510,
                        'LRG2': 0.706,
                        'LRG3': 0.930,
                        'ELG': 1.317,
                        'QSO': 1.491
                        }


tracer_params = {index: all_tracer_params[tracer] for index, tracer in enumerate(tracers)}
tracer_redshifts = {tracer: all_tracer_redshifts[tracer] for tracer in tracers}


# In[4]:


cosmo = Cosmoprimo(engine='class')
cosmo.init.params['H0'] = dict(derived=True)
cosmo.init.params['Omega_m'] = dict(derived=True)
cosmo.init.params['sigma8_m'] = dict(derived=True) 
# cosmo.init.params['fR0'] = dict(derived=False, latex ='f_{R0}')
#cosmo.init.params['theta_star'] = dict(derived=True)
fiducial = DESI() #fiducial cosmologyy


#Update cosmo priors
for param in ['n_s', 'h', 'omega_cdm', 'omega_b', 'logA', 'tau_reio']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            #cosmo.params[param].update(fixed = True)
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}) #ns10 planck
            # cosmo.params[param].update(prior={'dist': 'uniform','limits': [0.6, 1.3]}) #ns10 planck
    if param == 'omega_b': 
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': 0.00055}) #bbn prior
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.2, 1.]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.01, 0.99]})
    if param == 'm_ncdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.0, 5]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [1.61, 3.91]})
    # if param == 'w0_fld':
    #     cosmo.params[param].update(prior = {'dist':'uniform','limits': [-3, 1]})
    #     # cosmo.params[param].update(fixed=True)
    # if param == 'wa_fld':
    #     cosmo.params[param].update(prior = {'dist':'uniform','limits': [-3, 2]})    
        # cosmo.params[param].update(fixed=True)    
    if param == 'theta_star':
        cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.0104110, 'scale': 0.0000031})
        cosmo.params[param].update(fixed = False)
        cosmo.params[param].update(ref={'limits': [0.010407899999999999, 0.0104141]})
    # if param == 'fR0':
    #     if model != 'HS':
    #         print('model LCDM')
    #         cosmo.params[param].update(fixed=True, value=0.0)
    #     else:
    #         print('model HS')
    #         cosmo.params[param].update(
    #         prior={'dist': 'uniform', 'limits': [0, 9e-5]},
    #         fixed=False,
    #         ref={'limits': [0, 9e-5]}
    #         )


# In[ ]:


#Initialize an empty list to store the theory objects
theories = []

#Iterate over each tracer and create the corresponding theory object
all_tracer_sigma8_fid = {
                        'BGS': 0.69376997,
                        'LRG1': 0.6208944,
                        'LRG2': 0.56378872,
                        'LRG3': 0.50823223,
                        'ELG': 0.43197292,
                        'QSO': 0.4041147
                        }
for tracer in tracers:
    if tracer in tracer_redshifts:
        z = tracer_redshifts[tracer]
        sigma8_fid = all_tracer_sigma8_fid[tracer]
    else:
        print(f'Invalid tracer: {tracer}. Skipping.') 
        continue

    #Create the template and theory objects
    template = DirectPowerSpectrumTemplate(fiducial = fiducial,cosmo = cosmo, z=z) #cosmology and fiducial cosmology defined above

    
    # Activate the corresponding model
    if pt_model == "lpt_velocileptors":
        print("LPTVelocileptorsTracerPowerSpectrumMultipoles activated")
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis=prior_basis)
    elif pt_model == "rept_velocileptors":
        print("REPTVelocileptorsTracerPowerSpectrumMultipoles activated")
        theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis=prior_basis)
    elif pt_model == "ept_velocileptors":
        print("EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles activated")
        theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis=prior_basis)
    elif pt_model == "pybird":
        print("PyBirdPowerSpectrumMultipoles activated")
        theory = PyBirdPowerSpectrumMultipoles(template=template, prior_basis=prior_basis)
    else:
        print("No valid pt_model specified, defaulting to FOLPS model")
        theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, prior_basis=prior_basis,A_full=False, b3_coev=True) #Add the prior_basis='physical' argument to use physically motivated priors

    
    #Update bias and EFT priors
    #theory.params['bs'].update(fixed=True)
    #Update bias and EFT priors
    #theory.params['bs'].update(fixed=True)
    if prior_basis == 'physical':
        theory.params['b1p'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
        theory.params['b2p'].update(prior = {'dist':'uniform','limits': [-50, 50]})
        theory.params['bsp'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 40})
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


# In[ ]:


#Define a function to create an observable
def create_observable(data_fn, wmatrix_fn, covariance_fn, theory, tracer_index,k_max):
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
observables = [create_observable(params['data_fn'], params['wmatrix_fn'], params['covariance_fn'],theories, all_tracer_indices[tracer],all_tracer_kmax[tracer]) 
                for tracer, params in all_tracer_params.items()]


# In[ ]:


if set_emulator:
    #Create an emulated theory for each tracer
    for i in range(len(theories)):
        k_max=k_max_array[i]
    
        # suffix = f"_kmax{k_max}" if k_max > 0.301 else ""
        suffix = f"_kmax{0.39}" if k_max > 0.301 else ""
        emulator_filename = (
    f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/'
    f'Emulator_y1/emu_-{tracers_str}_{tracers[i]}{suffix}.npy'
)
            
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


# In[ ]:


#Analytic marginalization over eft and nuisance parameters
for i in range(len(theories)): 
    if prior_basis == 'physical':
        params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
        params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for param in params_list:    
        theories[i].params[param].update(derived = '.marg')
        # theories[i].params[param].update(derived = '.best') #Jeffrey   
        
#Rename the eft and nuisance parameters to get a parameter for each tracer (i.e. QSO_alpha0, QSO_alpha2, BGS_alpha0,...)        
for i in range(len(theories)):    
    for param in theories[i].init.params:
        # Update latex just to have better labels
        param.update(namespace='{}'.format(tracers[i]))   
        
# for i in range(len(theories)): 
#     params_list = ['w0_fld','wa_fld']
#     for param in params_list:    
#         theories[i].pt.params[param].update(fixed=True) 

# In[ ]:

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

    
# chain_path = Path(f'{chain_name}.npy')
# chain_path=Path('/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains/base_QSO_0.351/MCMC-fs_klim_0-0.02-0.301_2-0.02-0.301-all-GCcomb_schoneberg2024-bbn_planck2018-ns10_physical-prior_LCDM_folpsD_Ano_tight_eft.npy')
# chain = load_chain(chain_path)
# likelihood(**chain.choice(index='argmax', input=True))
# observables[-1].plot(fn='test_bestfit.png',kw_save={'dpi':250})
# theories[-1].plot(fn='test_bestfit_theory.png',kw_save={'dpi':250})

# In[ ]:


#Run the sampler and save the chain
from desilike.samplers import EmceeSampler, MCMCSampler

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
