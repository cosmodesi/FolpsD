#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.insert(0, "/global/u1/p/prakharb/desilike")
#sys.path.insert(0, "/global/u1/p/prakharb/desilike")
#sys.path.insert(0, "/global/u1/p/prakharb/cosmoprimo")
import desilike, inspect
print(inspect.getfile(desilike))


import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles,  DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles
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
from desilike_mcmc.mike_data_tools import *


# In[ ]:


######### Settings #########

#model: LCDM or HS
model = 'LCDM'

#Put 'True' to resume chain. 'False' to start from 0 steps
restart_chain = False
#Biasing and EFT parametrization: 'physical' or 'default' (non-physical)
prior_basis = 'standard' #Prior to be used 

k_max = 0.14

#width for EFT and SN paramss
width_EFT = 12.5
width_SN0 = 2.0
width_SN2 = 5.0

pt_model = "folps"  # Change this to "rept_velocileptors" or "pybird" as needed

sampler = 'cobaya'

set_emulator = True

GR_criteria = 0.01  # R - 1 < GR_criteria 


# Argument parser
#parser = argparse.ArgumentParser(description="Run MCMC analysis with different tracers")
#parser.add_argument("--tracers", nargs="+", required=True, help="List of tracers to use")
#args = parser.parse_args()


# List of tracers
# tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']  # Add more tracers as needed
tracers = ['LRG2']
#tracers = args.tracers
all_tracers = {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}

if set(tracers) == all_tracers:
    tracers_str = "all"
else:
    tracers_str = "+".join(tracers)

chain_name = f'chains/MCMC-fs_abacus2gen_folpsv2_bispectrum_test_0.20_0.14_var_nuis_uniform_0.2cov'
if not set_emulator:
    chain_name += '_noemu'
#print(chain_name)


# Define file paths for each tracer (in the same order as the tracers list above)
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
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.5,0.9]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.05, 0.2]})
    #if param == 'm_ncdm':
        #cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.0, 5]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})
    #if param == 'fR0':
    #    if model != 'HS':
    #        print('model '+str(model))
    #        cosmo.params[param].update(fixed=True, value=0.0)
    #    else:
    #        print('model HS')
    #        cosmo.params[param].update(
    #        prior={'dist': 'uniform', 'limits': [0, 9e-5]},
    #        fixed=False,
    #        ref={'limits': [0, 9e-5]}
    #        )

#Define tracer types and their corresponding redshifts
all_tracer_redshifts = {
                        'BGS': 0.295,
                        'LRG1': 0.510,
                        'LRG2': 0.8,
                        'LRG3': 0.930,
                        'ELG': 1.317,
                        'QSO': 1.491
                        }

tracer_params = {index: all_tracer_params[tracer] for index, tracer in enumerate(tracers)}
tracer_redshifts = {tracer: all_tracer_redshifts[tracer] for tracer in tracers}

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
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis="physical")
    elif pt_model == "rept_velocileptors":
        print("REPTVelocileptorsTracerPowerSpectrumMultipoles activated")
        theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis="physical")
    elif pt_model == "ept_velocileptors":
        print("EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles activated")
        theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis="physical")
    elif pt_model == "pybird":
        print("PyBirdPowerSpectrumMultipoles activated")
        theory = PyBirdPowerSpectrumMultipoles(template=template, prior_basis="physical")
    else:
        print("No valid pt_model specified, defaulting to FOLPS model")
        theory = FOLPSv2TracerBispectrumMultipoles(template=template, prior_basis="standard") #Add the prior_basis='physical' argument to use physically motivated priors

    #Update bias and EFT priors
    #theory.params['bs'].update(fixed=True)
    if prior_basis == 'physical':
        theory.params['b1p'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
        theory.params['b2p'].update(prior = {'dist':'uniform','limits': [-50, 50]})
        theory.params['bsp'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 20})
        theory.params['alpha0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['alpha4p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
        theory.params['sn0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN0})
        theory.params['sn2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN2})
        #theory.params['X_FoG_p'].update(prior = {'dist':'uniform','limits': [0, 10]})
    else:
        theory.params['b1'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
        theory.params['b2'].update(prior = {'dist':'uniform','limits': [-50, 50]})
        theory.params['bs'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 20})
        for param in ['c1', 'c2','Pshot', 'Bshot', 'X_FoG_b']:
            theory.params[param].update(fixed = False)
        theory.params['c1'].update(prior={'dist':'uniform','limits': [-2000, 2000]})
        theory.params['c2'].update(prior={'dist':'uniform','limits': [-2000, 2000]})
        theory.params['Pshot'].update(prior={'dist':'uniform','limits': [-50000, 50000]})
        theory.params['Bshot'].update(prior={'dist':'uniform','limits': [-50000, 50000]})
        theory.params['X_FoG_b'].update(prior={'dist':'uniform','limits': [0, 15]})
    print("X_FoG_b=",theory.params['X_FoG_b'].value)
    #Append the theory object to the list
    theories.append(theory)

#Print used priors
for param in theory.all_params:
    print(param,':',theory.all_params[param].prior)

#Define data vector and covariance using mike data tools

def load_data_vector_and_covariance():
    k_min=0.02
    k_max=0.201
    k_max_b0 = 0.20
    k_max_b2 = 0.14

    isP0, isP2, isP4 =False, False, False
    isB000, isB202 = True, True

    Vol=1

    tracer='LRG'
    z_str='z0.800'
    z_evaluation=0.8

    path_fits='chains/'

    now = datetime.now()
    tiempo=now.strftime("%m-%d-%Y-%H%M")

    name=f"c_FolpsD__{tracer}_z{z_evaluation:.3f}_Pkmax-{k_max:.3f}_B000kmax-{k_max_b0:.3f}_B202kmax-{k_max_b2:.3f}_bsfree"


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

    # Read covariance

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
    # cov_inv_arr = np.linalg.inv(cov_arr)
    
    # kr_pk = np.load("./desilike_mcmc/kr_pk.npy")
    # data = np.load("./desilike_mcmc/data.npy")
    # cov_arr = np.load("./desilike_mcmc/cov.npy")
    np.save("./desilike_mcmc/kr_b0.npy",kr_b0)
    np.save("./desilike_mcmc/data_b0.npy",data)
    np.save("./desilike_mcmc/cov_b0.npy",cov_array)
    return data, cov_array, kr_b0, kr_b2

def load_data_vector_and_covariance_files():
    kr_pk = np.load("./desilike_mcmc/kr_pk.npy")
    data = np.load("./desilike_mcmc/data.npy")
    cov_arr = np.load("./desilike_mcmc/cov.npy")
    return data, cov_arr, kr_pk


#Define a function to create an observable
def create_observable( #wmatrix_fn, covariance_fn,
                      theory, tracer_index):
   
    data, cov_arr, kr_b0, kr_b2 = load_data_vector_and_covariance()
    cov_arr = 0.2*cov_arr
    print(data.shape,cov_arr.shape,kr_b0.shape,kr_b2.shape)
    
    #Create and return the observable: following  https://desi.lbl.gov/trac/wiki/keyprojects/y1kp3/clusteringproducts#a101:Iwanttowritemychi2
    return TracerPowerSpectrumMultipolesObservable(
        data=data,
        covariance=cov_arr,
        theory=theories[tracer_index],
        ells = [0,2],
        k = [kr_b0,kr_b2]
    )
#Create observables for each tracer
observables = [create_observable(
                                 #params['wmatrix_fn'], params['covariance_fn'],
                                 theories, i) 
                for i, params in tracer_params.items()]


if set_emulator:
    # Create an emulated theory for each tracer
    for i in range(len(theories)):
        emulator_filename = f'Emulator_test_sims_bispectrum/emu_-{tracers_str}_{str(tracers[i])}_0.20_0.14.npy'
        
        if os.path.exists(emulator_filename):
            print(f"FS emulator {i} already exists, loading it now")
            emulator = EmulatedCalculator.load(emulator_filename)  
            theories[i].init.update(pt=emulator)
         
        else:
            theories[i] = observables[i].wmatrix.theory
            emulator = Emulator(theories[i].pt,
                                 engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulated_pt = emulator.to_calculator()
            emulated_pt.save(emulator_filename)
            theories[i].init.update(pt=emulated_pt)
            

print('All theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED, proceeding without emulation')

#Analytic marginalization over eft and nuisance parameters
# for i in range(len(theories)): 
#     if prior_basis == 'physical':
#         params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
#     else:
#          params_list = ['c1', 'c2', 'Pshot', 'Bshot']

#     for param in params_list:    
#         theories[i].params[param].update(derived = '.marg')

        
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

likelihood()

#Run the sampler and save the chain
from desilike.samplers import EmceeSampler, MCMCSampler

if sampler == 'cobaya':
    if restart_chain is False:
        sampler = MCMCSampler(likelihood, save_fn = chain_name)
        sampler.run(check={'max_eigen_gr': GR_criteria})
    else:
        sampler = MCMCSampler(likelihood ,save_fn = chain_name, 
                              chains=f'{chain_name}.npy')
        sampler.run(check={'max_eigen_gr': GR_criteria})
    
else:
    if restart_chain is False:
        sampler = EmceeSampler(likelihood ,save_fn = chain_name)
        sampler.run(check={'max_eigen_gr': GR_criteria})
    else:
        sampler = EmceeSampler(likelihood ,save_fn = chain_name, 
                               chains=f'{chain_name}.npy')
        sampler.run(check={'max_eigen_gr': GR_criteria})