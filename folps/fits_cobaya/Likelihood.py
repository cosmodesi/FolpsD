# file: likelihood/FolpsLikelihood.py
import numpy as np
from cobaya.likelihood import Likelihood
import FOLPSD as FOLPS
from cosmo_bacco import run_bacco
from mike_data_tools import ExtractDataAbacusSummit, ExtractDataEZmock, covariance, pole_k_selection

# ------------------------------
# Precompute data and covariance
# ------------------------------


# derived_params = ['sigma8', 'Omega_m']

k_min=0.02
k_max=0.201
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



# Load Abacus data
k_data_2nd, pkl0_2nd_all, pkl2_2nd_all, a, B000_2nd_all, B202_2nd_all = ExtractDataAbacusSummit(
    tracer, z_str, subtract_shot=True
)

Pk_0_2nd = np.mean(pkl0_2nd_all, axis=0)
Pk_2_2nd = np.mean(pkl2_2nd_all, axis=0)
B000_2nd = np.mean(B000_2nd_all, axis=0)
B202_2nd = np.mean(B202_2nd_all, axis=0)

# Load covariance
k_eff_all, pkl0ezmocks, pkl2ezmocks, pkl4ezmocks, B000ezmocks, B202ezmocks = ExtractDataEZmock(tracer, z_str)
k_cov_all, mean_ezmocks_all, cov_array_all = covariance(
    k_eff_all, pkl0ezmocks, pkl2ezmocks, pkl4ezmocks, B000ezmocks, B202ezmocks, Nscaling=Vol
)

pole_selection=[isP0, isP2, isP4,isB000, isB202]
print(pole_selection)

kmin_pk=0.02; kmax_pk=k_max
kmin_bk=0.02; 
ranges=[[kmin_pk,kmax_pk],[kmin_pk,kmax_pk],[kmin_pk,kmax_pk],[kmin_bk,k_max_b0],[kmin_bk,k_max_b2]]

mask=pole_k_selection(k_cov_all,pole_selection,ranges)
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


# ------------------------------
# Define Cobaya Likelihood Class
# ------------------------------
class FolpsLikelihood(Likelihood):
    """
    Cobaya likelihood wrapping the FOLPS-D model.
    """

    def initialize(self):
        # optional: cache things here
        self.z_evaluation = z_evaluation
        self.Omfid = 0.31519186799
        # self.derived_params = ["sigma8", "Omega_m"]
        FOLPS.Matrices()

    def get_requirements(self):
        # no external requirements (CAMB/CLASS), fully self-contained
        return {}

    def logp(self, **params):
        """
        Compute log-likelihood given parameters from Cobaya.
        """
        h = params["h"]
        omega_cdm = params["omegac"]
        omega_b = params["omegab"]
        logA_s = params["logAs"]
        b1, b2, bs2 = params["b1"], params["b2"], params["bs2"]
        # c1, c2 = params["c1"], params["c2"]
        # Pshot, Bshot = params["Pshot"], params["Bshot"]
        a_vir, a_vir_bk = params["a_vir"], params["a_vir_bk"]

        # --- Call your Folps model ---
        model = self._Folps(h, omega_cdm, omega_b, logA_s,
                            b1, b2, bs2, a_vir, a_vir_bk,k_thy)

        Pl02_const = model["pl02_const"]
        Pl02_i = model["pl02_i"]

        # Compute marginalized likelihood
        L0 = FOLPS.compute_L0(Pl_const=Pl02_const, Pl_data=data, invCov=cov_inv_arr)
        L1i = FOLPS.compute_L1i(Pl_i=Pl02_i, Pl_const=Pl02_const, Pl_data=data, invCov=cov_inv_arr)
        L2ij = FOLPS.compute_L2ij(Pl_i=Pl02_i, invCov=cov_inv_arr)

        invL2ij = np.linalg.inv(L2ij)
        detL2ij = np.linalg.det(L2ij)

        term1 = FOLPS.startProduct(L1i, L1i, invL2ij)
        term2 = np.log(abs(detL2ij))

        loglike = L0 + 0.5 * term1 - 0.5 * term2

        

        return loglike #, {'sigma8': sigma8, 'Omega_m': Omega_m}

    def _Folps(self, h, omega_cdm, omega_b, logA_s, b1, b2, bs2,
                a_vir, a_vir_bk,k_ev):
        """
        Wraps your Folps function but as a method.
        Stores sigma8 and Omega_m internally for output.
        """
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
        alpha0 = 20;                    #only for reference - does not affect the final result
        alpha2 = -58.8;                 #only for reference - does not affect the final result 
        alpha4 = 0.0;                       
        ctilde = 0.0;    
        alphashot0 = -0.073;            #only for reference - does not affect the final result
        alphashot2 = -6.38;             #only for reference - does not affect the final result    
        PshotP = 1/0.0002118763;    ##1/0.00017    ### it is completely degenerate with alphashot0
        
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
        
        Omfid=self.Omfid = 0.31519186799
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
    
        #pklir=FOLPS.pklIR_ini(LoopCorrections[0][0], LoopCorrections[0][1], LoopCorrections[1][1], h=h)
        k_pkl_pklnw=np.array([LoopCorrections[0][0], LoopCorrections[0][1], LoopCorrections[1][1]])
    
        
       
    
        
        
        # precision=[5,7,7]
        if isB000:
            bisp_nuis_params = [b1, b2, bs2, c1,c2,Pshot,Bshot, a_vir_bk]
            bisp_cosmo_params = [(omega_cdm+omega_b+omega_ncdm)/h**2,h]
    
            B000__, B202__ = FOLPS.Bisp_Sugiyama(bisp_cosmo_params, bisp_nuis_params,
                                                 k_pkl_pklnw=k_pkl_pklnw, z_pk=z_pk, k1k2pairs=k_ev_bk,
                                                 Omfid=Omfid)#, 
                                                   #, Omfid=-1,precision=precision)
            B000_ = FOLPS.interp(k_thy,k_ev_bk[:,0],B000__)
            B202_ = FOLPS.interp(k_thy,k_ev_bk[:,0],B202__)
        
            B000_const_mbin = m_bin @ B000_ 
            B202_const_mbin = m_bin @ B202_
        
            Pl02_const_binning = np.concatenate((Pkl0_const_mbin[k_points_pk], Pkl2_const_mbin[k_points_pk],
                                                 B000_const_mbin[k_points_b0], B202_const_mbin[k_points_b2])) 
        
        
        return ({'pl02_const':Pl02_const_binning, 'pl02_i':Pl02_i_binning})
