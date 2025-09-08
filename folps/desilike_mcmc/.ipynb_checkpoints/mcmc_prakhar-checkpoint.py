import sys
sys.path.insert(0, "/global/u1/p/prakharb/desilike")
import sys
sys.path.insert(0, "/global/u1/p/prakharb/cosmoprimo")
sys.path.insert(0,"/global/u1/p/prakharb/FOLPSpipe/folps")
import desilike, cosmoprimo, folps


import sys, os, shutil
import time
import emcee
import numpy as np
from schwimmbad import MPIPool

from datetime import datetime

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

# numberofpk0points=len(Pk_0_2nd[k_points_pk])
# numberofbk0points=len(B000_2nd[k_points_b0])
# numberofbk2points=len(B202_2nd[k_points_b2])

# cov_array=cov_array_all[np.ix_(mask, mask)]
# totsim = 2000 #number of sims
# n_data = len(data)
# Hartlap = (totsim - 1.) / (totsim - n_data - 2.)
# Hartlap
# cov_arr = cov_array * Hartlap
# cov_inv_arr = np.linalg.inv(cov_arr)


data = np.load("data.npy")
cov_arr = np.load("cov.npy")


from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles, KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate,FixedPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.full_shape import  FOLPSv2TracerPowerSpectrumMultipoles,FOLPSAXTracerPowerSpectrumMultipoles, PyBirdTracerPowerSpectrumMultipoles
from desilike.observables import ObservableCovariance
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine
from desilike import setup_logging

setup_logging()

template = DirectPowerSpectrumTemplate(z=0.8)
theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, tracer='LRG',freedom='max',kernels='fk',prior_basis='standard',damping='lor',k=kr_pk,ells=[0,2])
# emulator=EmulatedCalculator.load("_tests/folpsv2_emulator_LRG.npy")
observables = TracerPowerSpectrumMultipolesObservable(data=data,covariance=cov_arr,ells=[0,2],k=kr_pk,theory=theory)
likelihood = ObservablesGaussianLikelihood(observables=[observables])
# likelihood()


emulator=Emulator.load('_tests/folpsv2_emulator_LRG.npy')
theory=emulator.to_calculator()
observables = TracerPowerSpectrumMultipolesObservable(data=data,covariance=cov_arr,ells=[0,2],k=kr_pk,theory=theory)
likelihood = ObservablesGaussianLikelihood(observables=[observables])

params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

for param in params_list:    
   theory.params[param].update(derived = '.marg')

from desilike.samplers import ZeusSampler, EmceeSampler

# Let's just update the observable's theory, no need to redefine the observable & likelihood
# (Internally the code will reinitialize all calculators that depend on observable)


sampler = EmceeSampler(likelihood, save_fn='_tests/chain_fs_direct_*.npy', seed=42, chains='_tests/chain_fs_direct_*.npy')
sampler.run(check={'max_eigen_gr': 0.01})

