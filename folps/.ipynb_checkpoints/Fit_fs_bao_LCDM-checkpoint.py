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
model = 'w0waCDM'

#Put 'True' to resume chain. 'False' to start from 0 steps
restart_chain = True

#Biasing and EFT parametrization: 'physical' or 'default' (non-physical)
prior_basis = 'physical' 

#max. wavenumber 
k_max = 0.201

#width for EFT and SN paramss
width_EFT = 125
width_SN0 = 20
width_SN2 = 50

#PT models: 'folps', 'pybird', 'lpt_velocileptors', 'rept_velocileptors'
pt_model = "EFT" 

#samplers: 'cobaya', 'emcee'
sampler = 'cobaya'

# Set to 'True' to use emulators, or 'False' to disable them.
set_emulator = True

# R - 1 < GR_criteria 
GR_criteria = 0.05 


# In[3]:


# List of tracers
tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']  # Add more tracers as needed
all_tracers = {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}

if set(tracers) == all_tracers:
    tracers_str = "all"
else:
    tracers_str = "-".join(tracers)

chain_name = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains/base/MCMC-fs+bao_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_newfolpsv2_Ano_wide_eft'
if not set_emulator:
    chain_name += '_noemu'
#print(chain_name)


# Define file paths for each tracer (in the same order as the tracers list above)
all_tracer_params =  {
    'BGS': {
        'data_fn':'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.npy'
    },
    'LRG1': {
        'data_fn':'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.4-0.6.npy'
    },
    'LRG2': {
        'data_fn':'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.6-0.8.npy'
    },
    'LRG3': {
        'data_fn':'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.8-1.1.npy'
    },
    'ELG': {
        'data_fn':'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_ELG_LOPnotqso_GCcomb_z1.1-1.6.npy'
    },
    'QSO': {
        'data_fn':'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_QSO_GCcomb_z0.8-2.1.npy'
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
for param in ['n_s', 'h', 'omega_cdm', 'omega_b', 'logA', 'tau_reio', 'w0_fld','wa_fld']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            #cosmo.params[param].update(fixed = True)
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}) #ns10 planck
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
    if param == 'w0_fld':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [-3, 1]})
        # cosmo.params[param].update(fixed=True)
    if param == 'wa_fld':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [-3, 2]})    
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
for tracer in tracers:
    if tracer in tracer_redshifts:
        z = tracer_redshifts[tracer]
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
    if prior_basis == 'physical':
        theory.params['b1p'].update(prior = {'dist':'uniform','limits': [0, 3]})
        theory.params['b2p'].update(prior = {'dist':'norm','loc': 0, 'scale': 5 })
        theory.params['bsp'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 5})
        theory.params['alpha0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha4p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['sn0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN0})
        theory.params['sn2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN2})
        theory.params['X_FoG_pp'].update(fixed=True)
        # theory.params['X_FoG_pp'].update(prior = {'dist':'uniform','limits': [0, 10]})
    else:
        theory.params['b1'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
        theory.params['b2'].update(prior = {'dist':'uniform','limits': [-50, 50]})
        theory.params['bs'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 20})
        theory.params['alpha0'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha2'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha4'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['sn0'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN0})
        theory.params['sn2'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN2})
        theory.params['X_FoG_p'].update(fixed=True)
        # theory.params['X_FoG_p'].update(prior = {'dist':'uniform','limits': [0, 10]})
    
    #Append the theory object to the list
    theories.append(theory)


# In[ ]:


#Define a function to create an observable
def create_observable(data_fn, theory, tracer_index):
    #Load and process covariance
    covariance = ObservableCovariance.load(data_fn)
    #covariance = covariance.select(xlim=(0.02, k_max), projs=[0, 2])
    
    #power spectrum
    data = covariance.observables('power')

    indices = np.where((data.flatx > 0.02) & (data.flatx < k_max))[0]  #new-line
    
    #Create and return the observable: following  https://desi.lbl.gov/trac/wiki/keyprojects/y1kp3/clusteringproducts#a101:Iwanttowritemychi2
    return TracerPowerSpectrumMultipolesObservable(
        data=data,
        covariance=covariance.select(observables=data, select_observables=True),
        klim={ell: [0.02, k_max, 0.005] for ell in [0, 2]},
        theory=theories[tracer_index],
        kin=data.attrs['kin'],
        wmatrix=data.attrs['wmatrix'][indices, :],
        ellsin=data.attrs['ellsin'],
        wshotnoise=data.attrs['wshotnoise'][indices]
    )

#Create observables for each tracer
observables = [create_observable(params['data_fn'], theories, i) 
                for i, params in tracer_params.items()]


# In[ ]:


if set_emulator:
    #Create an emulated theory for each tracer
    for i in range(len(theories)):
        emulator_filename = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Emulator_y1_EFT/emu_{model}-{tracers_str}_{str(tracers[i])}.npy'
        
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


flatdatas=[]

def create_bao_observable(data_fn, tracer_index):
    # Create and return the observable
    forfit = ObservableCovariance.load(data_fn)
    covariance, data = forfit.view(observables='bao-recon'), forfit.observables(observables='bao-recon')   
    observable = BAOCompressionObservable(data=data.view(), covariance=covariance, cosmo=cosmo, quantities=data.projs, fiducial=fiducial, z=data.attrs['zeff'])
    flatdata = observable.flatdata
    flatdatas.append(flatdata)
    #observable.init.update(cosmo=cosmo)
    return observable
    
bao_observables = [create_bao_observable(params['data_fn'], i) 
                for i, params in tracer_params.items()]


# In[ ]:


if set_emulator:
    for i in range(len(bao_observables)):
        emulator_filename = f'/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Emulator_y1_bao_EFT/emu-bao_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_{str(tracers[i])}.npy'
        
        if os.path.exists(emulator_filename):
            print(f"BAO emulator {i} already exists, loading it now")
            emulator = EmulatedCalculator.load(emulator_filename)  
            bao_observables[i] = emulator
        else:
            print(f" Computing BAO emulator {i}")
            emulator = Emulator(bao_observables[i], engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulator.save(emulator_filename)
            bao_observables[i] = emulator.to_calculator()   
        for p in ['w0_fld', 'wa_fld']:
            bao_observables[i].params[p].update(fixed=True, value={'w0_fld': -1.0, 'wa_fld': 0.0}[p])
print('BAO has been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED for BAO; proceeding without emulation')


# In[ ]:

setup_logging()
Likelihoods = []
for i in range(len(observables)):
    forfit = ObservableCovariance.load(tracer_params[i]['data_fn'])
    
    data = forfit.observables('power')
    
    indices = np.where((data.flatx > 0.02) & (data.flatx < k_max))[0]
    size_pk = 72
    size_baoxbao = forfit.view().shape[0] - size_pk
    new_indices = indices.tolist() + list(range(size_pk, size_pk + size_baoxbao))
    
    covariance = forfit.view()
    cov = covariance[np.ix_(new_indices, new_indices)]

    #print(size_pk)
    #print(cov.shape)
    likely = ObservablesGaussianLikelihood(observables = [observables[i],bao_observables[i]], covariance = cov)
    likely()
    Likelihoods.append(likely)

likelihood = SumLikelihood(Likelihoods)
print("likelihood: ",likelihood())
print("Test Successful!")

    
#Create a likelihood per theory object
# setup_logging()
# Likelihoods = []
# for i in range(len(observables)):
#    forfit = ObservableCovariance.load(tracer_params[i]['data_fn'])
#    covariance = forfit.view()
#    likely = ObservablesGaussianLikelihood(observables = [observables[i],bao_observables[i]], covariance = covariance)
#    likely()
#    Likelihoods.append(likely)


# Run the sampler and save the chain
from desilike.samplers import EmceeSampler, MCMCSampler

from pathlib import Path
def load_chain(fi, burnin=0.3):
    from desilike.samples import Chain
    # chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
    chains = [Chain.load(fi).remove_burnin(burnin)]
    chain = chains[0].concatenate(chains)
    print(f'chain: {chain}')
    return chain

cov_chain_path = Path('/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/y1_chains/EFT/folps_chains/MCMC-fs+bao_klim_0-0.02-0.2_2-0.02-0.2-all-GCcomb_schoneberg2024-bbn_planck2018-ns10_physical-prior_LCDM_folps.npy')
cov_chain = load_chain(cov_chain_path,burnin=0.3)
if sampler == 'cobaya':
    if restart_chain is False:
        sampler = MCMCSampler(sum(Likelihoods), save_fn = chain_name)
        sampler.run(check={'max_eigen_gr': GR_criteria})
    else:
        sampler = MCMCSampler(sum(Likelihoods) ,save_fn = chain_name,
                              chains=f'{chain_name}.npy')
        sampler.run(check={'max_eigen_gr': GR_criteria})
    
else:
    if restart_chain is False:
        sampler = EmceeSampler(sum(Likelihoods) ,save_fn = chain_name)
        sampler.run(check={'max_eigen_gr': GR_criteria})
    else:
        sampler = EmceeSampler(sum(Likelihoods) ,save_fn = chain_name, 
                               chains=f'{chain_name}.npy')
        sampler.run(check={'max_eigen_gr': GR_criteria})

