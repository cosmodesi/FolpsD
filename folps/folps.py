#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================================================ #
#                                             FOLPS                                            #
# ============================================================================================ #
#     Fast and Efficient Computation of the Redshift-Space Power Spectrum and Bispectrum       #
#         in Cosmological Models with Massive Neutrinos and Modified Gravity Theories          #
# -------------------------------------------------------------------------------------------- #
#     Designed for high-precision, high-performance modeling in state-of-the-art LSS           #
#     analyses. FOLPS delivers robust predictions across a wide range of theoretical models.   #
#                                                                                              #
#     “Simple things should be simple. Complex things should be possible.”                     #
#                                — Alan Kay                                                    #
#     FOLPS makes them both effortless.                                                        #
# ============================================================================================ #


# In[2]:


import os
import scipy
from scipy import special
from scipy import integrate

# Global variable to store the preferred backend (default: 'numpy')
PREFERRED_BACKEND = os.environ.get("FOLPS_BACKEND", "numpy")  #options:"numpy" & "jax" 

class BackendManager:
    def __init__(self, preferred_backend='numpy'):
        """Initializes the backend according to the user's preference."""
        #self.using_jax = False
        self.backend = None
        self.modules = {}  # Stores modules for dynamic usage
        self.setup_backend(preferred_backend)

    def setup_backend(self, preferred_backend='numpy'):
        """Dynamically configures the backend between NumPy and JAX."""
        if preferred_backend == 'jax':
            try:
                import jax
                if any(device.device_kind == "Gpu" for device in jax.devices()):
                    print("✅ GPU detected. Using JAX with GPU.")
                else:
                    print("⚠️ No GPU found. Using JAX with CPU.")

                from jax import numpy as np
                from tools_jax import interp, simpson, legendre, extrapolate, extrapolate_pklin, get_pknow, get_linear_ir, get_linear_ir_ini
                self.modules = {
                    "np": np,
                    "interp": interp,
                    "simpson": simpson,
                    "legendre": legendre,
                    "extrapolate": extrapolate,
                    "extrapolate_pklin": extrapolate_pklin,
                    "get_pknow": get_pknow,
                    "get_linear_ir":get_linear_ir,
                    "get_linear_ir_ini":get_linear_ir_ini,
                }
                #self.using_jax = True
                self.backend = 'jax'
            except (RuntimeError, ImportError) as e:
                print(f"❌ Error initializing JAX: {e}")
                print("⏳ Falling back to NumPy...")
                self.setup_backend('numpy')
        elif preferred_backend == 'numpy':
            print("✅ Using NumPy with CPU.")
            import numpy as np
            from tools import interp, simpson, legendre, extrapolate, extrapolate_pklin, get_pknow, get_linear_ir, get_linear_ir_ini
            self.modules = {
                "np": np,
                "interp": interp,
                "simpson": simpson,
                "legendre": legendre,
                "extrapolate": extrapolate,
                "extrapolate_pklin": extrapolate_pklin,
                "get_pknow": get_pknow,
                "get_linear_ir":get_linear_ir,
                "get_linear_ir_ini":get_linear_ir_ini,
            }
            #self.using_jax = False
            self.backend = 'numpy'
        else:
            raise ValueError("⚠️ Invalid backend specified. Choose 'jax' or 'numpy'.")

    def get_module(self, name):
        """Retrieves a module or function from the current backend."""
        return self.modules.get(name, None)


# Initialize with JAX if available
backend_manager = BackendManager(PREFERRED_BACKEND)

# Access functions and modules
np = backend_manager.get_module("np")
interp = backend_manager.get_module("interp")
simpson = backend_manager.get_module("simpson")
legendre = backend_manager.get_module("legendre")
extrapolate = backend_manager.get_module("extrapolate")
extrapolate_pklin = backend_manager.get_module("extrapolate_pklin")
get_pknow = backend_manager.get_module("get_pknow")
get_linear_ir = backend_manager.get_module("get_linear_ir")
get_linear_ir_ini = backend_manager.get_module("get_linear_ir_ini")
backend = backend_manager.backend
#using_jax = backend_manager.using_jax


# In[3]:


def get_fnu(h, ombh2, omch2, omnuh2):
    """
    Gives some inputs for the function 'f_over_f0_EH'.

    Args:
        h = H0/100.
        ombh2: Omega_b h² (baryons)
        omch2: Omega_c h² (CDM)
        omnuh2: Omega_nu h² (massive neutrinos)
    Returns:
        h: H0/100.
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        fnu: Omega_nu/OmM0
        mnu: Total neutrino mass [eV]
    """
    Omb = ombh2 / h**2
    Omc = omch2 / h**2
    Omnu = omnuh2 / h**2

    OmM0 = Omb + Omc + Omnu
    fnu = Omnu / OmM0
    mnu = Omnu * 93.14 * h**2

    return(h, OmM0, fnu, mnu)


# In[4]:


def f_over_f0_EH(zev, k, OmM0, h, fnu, Nnu=3, Neff=3.046):
    """
    Routine to get f(k)/f0 and f0.
    f(k)/f0 is obtained following H&E (1998), arXiv:astro-ph/9710216
    f0 is obtained by solving directly the differential equation for the linear growth at large scales.

    Args:
        zev: redshift
        k: wave-number
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        h = H0/100
        fnu: Omega_nu/OmM0
        Nnu: number of massive neutrinos
        Neff: effective number of neutrinos
    Returns:
        f(k)/f0 (when 'EdSkernels = True' f(k)/f0 = 1)
        f0
    """
    #def interp(k, x, y):  # out-of-range below
    #    from scipy.interpolate import CubicSpline
    #    return CubicSpline(x, y)(k)
    
    eta = np.log(1 / (1 + zev))   #log of scale factor
    omrv = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff)) #rad: including neutrinos
    aeq = omrv/OmM0           #matter-radiation equality

    pcb = 5./4 - np.sqrt(1 + 24*(1 - fnu))/4     #neutrino supression
    c = 0.7
    theta272 = (1.00)**2                         # T_{CMB} = 2.7*(theta272)
    pf = (k * theta272)/(OmM0 * h**2)
    DEdS = np.exp(eta)/aeq                      #growth function: EdS cosmology

    fnunonzero = np.where(fnu != 0., fnu, 1.)
    yFS = 17.2*fnu*(1 + 0.488*fnunonzero**(-7/6))*(pf*Nnu/fnunonzero)**2  #yFreeStreaming
    # pcb = 0. and yFS = 0. when fnu = 0.
    rf = DEdS/(1 + yFS)
    fFit = 1 - pcb/(1 + (rf)**c)                #f(k)/f0

    #Getting f0
    def OmM(eta):
        return 1./(1. + ((1-OmM0)/OmM0)*np.exp(3*eta) )

    def f1(eta):
        return 2. - 3./2. * OmM(eta)

    def f2(eta):
        return 3./2. * OmM(eta)

    etaini = -6  #initial eta, early enough to evolve as EdS (D + \propto a)
    zfin = -0.99

    def etaofz(z):
        return np.log(1/(1 + z))

    etafin = etaofz(zfin)
    
    from scipy.integrate import odeint
    #if using_jax:
    if backend == 'jax':
        from jax.experimental.ode import odeint

    # differential eq.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return [Dprime, f2(eta)*Df - f1(eta)*Dprime]

    #eta range and initial conditions
    eta = np.linspace(etaini, etafin, 1001)   
    Df0 = np.exp(etaini)
    Df_p0 = np.exp(etaini)
        
    #solution
    Dplus, Dplusp = odeint(Deqs, [Df0,Df_p0], eta).T
    
    Dplusp_ = interp(etaofz(zev), eta, Dplusp)
    Dplus_ = interp(etaofz(zev), eta, Dplus)
    f0 = Dplusp_/Dplus_ 

    return (k, fFit, f0)


# In[5]:


def get_cm(kmin, kmax, N, b_nu, inputpkT):
    """
    Coefficients c_m, see eq.~ 4.2 - 4.5 at arXiv:2208.02791

    Args:
        kmin, kmax: minimal and maximal range of the wave-number k.
        N: number of sampling points (we recommend using N=128).
        b_nu: FFTLog bias (use b_nu = -0.1. Not yet tested for other values).
        inputpkT: k-coordinates and linear power spectrum.
    Returns:
        coefficients c_m (cosmological dependent terms).
    """
    #def interp(k, x, y):  # out-of-range below
    #    from scipy.interpolate import CubicSpline
    #    return CubicSpline(x, y)(k)
    
    #define de zero matrices
    M = int(N/2)
    k, pk = inputpkT
    ii = np.arange(N)

    #"kbins" trought "delta" gives logspaced k's in [kmin, kmax]
    kbins = kmin * np.exp(ii * np.log(kmax / kmin) / (N - 1))
    f_kl = interp(kbins, k, pk) * (kbins / kmin)**(-b_nu)

    #F_m is the Discrete Fourier Transform (DFT) of f_kl
    #"forward" has the direct transforms scaled by 1/N (numpy version >= 1.20.0)
    F_m = np.fft.fft(f_kl, n=N) / N

    #etaT = bias_nu + i*eta_m
    #to get c_m: 1) reality condition, 2) W_m factor
    ii = np.arange(N + 1)
    etaT = b_nu + (2*np.pi*1j/np.log(kmax/kmin)) * (ii - N/2) * (N-1) / N
    c_m = kmin**(-(etaT))*F_m[ii - M]
    c_m = np.concatenate([c_m[:1] / 2., c_m[1:-1], c_m[-1:] / 2.])

    return c_m


# In[6]:


class MatrixCalculator:
    """
    A class to compute M matrices that are independent of cosmological parameters, 
    and thus only need to be calculated once per instance.

    Parameters:
        nfftlog (int, optional): Number of sample points for FFTLog integration. Defaults to 128.
                                 It is recommended to use this default value for numerical accuracy; 
                                 see Figure 8 in arXiv:2208.02791.
        A_full (bool, optional): Whether to compute the full A_TNS function. If True (default), the 
                                 function includes contributions from b1, b2, and bs2. If False, it 
                                 uses an approximation based only on the linear bias b1.

    Notes:
        - The wavenumber range (k) is fixed internally to kmin = 1e-7 and kmax = 100.
        - The bias parameter b_nu is fixed to -0.1. Other values have not been tested.

    Returns:
        list: A list containing all computed M matrices.
    """
    def __init__(self, nfftlog=128, A_full=True, remove_DeltaP=False):
        global A_full_status, remove_DeltaP_status
        self.nfftlog = nfftlog
        self.kmin = 10**(-7)
        self.kmax = kmax=100.
        self.b_nu = -0.1  # not yet tested for other values
        self.A_full = A_full
        A_full_status = A_full
        
        self.remove_DeltaP = remove_DeltaP
        remove_DeltaP_status = remove_DeltaP
        
        self.filename = f'matrices_nfftlog{self.nfftlog}_Afull_{A_full_status}_remove-DeltaP_{remove_DeltaP_status}.npy'
    
        
        if remove_DeltaP:
            print("removing $\Delta P(k,\mu)$... WARNING: This violates momentum conservation!!!")
        
    def Imatrix(self, nu1, nu2):
        return 1 / (8 * np.pi**(3 / 2.)) * (special.gamma(3 / 2. - nu1) * special.gamma(3 / 2. - nu2) * special.gamma(nu1 + nu2 - 3 / 2.))\
                / (special.gamma(nu1) * special.gamma(nu2) * special.gamma(3 - nu1 - nu2))
    
    #M22-type
    def M22(self, nu1, nu2):
            
        #Overdensity and velocity
        def M22_dd(nu1, nu2):
            return self.Imatrix(nu1,nu2)*(3/2-nu1-nu2)*(1/2-nu1-nu2)*( (nu1*nu2)*(98*(nu1+nu2)**2 - 14*(nu1+nu2) + 36) - 91*(nu1+nu2)**2+ 3*(nu1+nu2) + 58)/(196*nu1*(1+nu1)*(1/2-nu1)*nu2*(1+nu2)*(1/2-nu2))
            
        def M22_dt_fp(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-23-21*nu1+(-38+7*nu1*(-1+7*nu1))*nu2+7*(3+7*nu1)*nu2**2) )/(196*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fpfp(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-12*(1-2*nu2)**2 + 98*nu1**(3)*nu2 + 7*nu1**2*(1+2*nu2*(-8+7*nu2))- nu1*(53+2*nu2*(17+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
            
        def M22_tt_fkmpfp(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-37+7*nu1**(2)*(3+7*nu2) + nu2*(-10+21*nu2) + nu1*(-10+7*nu2*(-1+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2))
        
        #A function
        def MtAfp_11(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-5+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*(-1+2*nu1)*nu2)
        
        def MtAfkmpfp_12(nu1, nu2):
            return -self.Imatrix(nu1,nu2)*(((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(6+7*(nu1+nu2)))/(56*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfp_22(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-18+3*nu1*(1+4*(10-9*nu1)*nu1)+75*nu2+8*nu1*(41+2*nu1*(-28+nu1*(-4+7*nu1)))*nu2+48*nu1*(-9+nu1*(-3+7*nu1))*nu2**2+4*(-39+4*nu1*(-19+35*nu1))*nu2**3+336*nu1*nu2**4) )/(56*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def MtAfpfp_22(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-5+3*nu2+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*nu2)
        
        def MtAfkmpfpfp_23(nu1, nu2):
            return -self.Imatrix(nu1,nu2)*(((-1+7*nu1)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(28*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfpfp_33(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-13*(1+nu1)+2*(-11+nu1*(-1+14*nu1))*nu2 + 4*(3+7*nu1)*nu2**2))/(28*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        #Some B functions, not called by default
        def MB2_21(nu1, nu2):
            return -2*((-15*self.Imatrix(-3 + nu1,2 + nu2))/64. + (15*self.Imatrix(-2 + nu1,1 + nu2))/16. + (3*self.Imatrix(-2 + nu1,2 + nu2))/4. - (45*self.Imatrix(-1 + nu1,nu2))/32. - (9*self.Imatrix(-1 + nu1,1 + nu2))/8. - (27*self.Imatrix(-1 + nu1,2 + nu2))/32. + (15*self.Imatrix(nu1,-1 + nu2))/16. + (3*self.Imatrix(nu1,1 + nu2))/16. + (3*self.Imatrix(nu1,2 + nu2))/8. - (15*self.Imatrix(1 + nu1,-2 + nu2))/64. + (3*self.Imatrix(1 + nu1,-1 + nu2))/8. - (3*self.Imatrix(1 + nu1,nu2))/32. - (3*self.Imatrix(1 + nu1,2 + nu2))/64.)
       
        def MB3_21(nu1, nu2):
            return -2*((35*self.Imatrix(-3 + nu1,2 + nu2))/128. - (35*self.Imatrix(-2 + nu1,1 + nu2))/32. - (25*self.Imatrix(-2 + nu1,2 + nu2))/32. + (105*self.Imatrix(-1 + nu1,nu2))/64. + (45*self.Imatrix(-1 + nu1,1 + nu2))/32. + (45*self.Imatrix(-1 + nu1,2 + nu2))/64. - (35*self.Imatrix(nu1,-1 + nu2))/32. - (15*self.Imatrix(nu1,nu2))/32. - (9*self.Imatrix(nu1,1 + nu2))/32. - (5*self.Imatrix(nu1,2 + nu2))/32. + (35*self.Imatrix(1 + nu1,-2 + nu2))/128. - (5*self.Imatrix(1 + nu1,-1 + nu2))/32. - (3*self.Imatrix(1 + nu1,nu2))/64. - self.Imatrix(1 + nu1,1 + nu2)/32. - (5*self.Imatrix(1 + nu1,2 + nu2))/128.)
        
        def MB2_22(nu1, nu2):
            return self.Imatrix(nu1, nu2)*(-9*(-3 + 2*nu1 + 2*nu2)*(-1 + 2*nu1 + 2*nu2)*(3 + 4*nu1**2 + nu1*(2 - 12*nu2) + 2*nu2 + 4*nu2**2))/(64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
        
        def MB3_22(nu1, nu2):
            return self.Imatrix(nu1, nu2)*(3*(-3 + 2*nu1 + 2*nu2)*(-1 + 2*nu1 + 2*nu2)*(1 + 2*nu1 + 2*nu2)*(3 + 4*nu1**2 + nu1*(2 - 12*nu2) + 2*nu2 + 4*nu2**2))/(64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))

        def MB4_22(nu1, nu2):
            return self.Imatrix(nu1, nu2)*((-3 + 2*nu1)*(-3 + 2*nu2)*(-3 + 2*nu1 + 2*nu2)*(-1 + 2*nu1 + 2*nu2)*(1 + 2*nu1 + 2*nu2)*(3 + 2*nu1 + 2*nu2))/(64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
        
        #D function
        def MB1_11(nu1, nu2):
            return self.Imatrix(nu1,nu2)*(3-2*(nu1+nu2))/(4*nu1*nu2)
        
        def MC1_11(nu1, nu2):
            if remove_DeltaP_status:
                return 0 * self.Imatrix(nu1,nu2)
            else:
                return self.Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*(nu1+nu2)))/(4*nu2*(1+nu2)*(-1+2*nu2))
        
        def MB2_11(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2)
        
        def MC2_11(nu1, nu2):
            if remove_DeltaP_status:
                return 0 * self.Imatrix(nu1,nu2)
            else:
                return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu2*(1+nu2))
        
        def MD2_21(nu1, nu2):
            if remove_DeltaP_status:
                return MB2_21(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((-1+2*nu1-4*nu2)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2*(-1+nu2+2*nu2**2))
        
        def MD3_21(nu1, nu2):
            if remove_DeltaP_status:
                return MB3_21(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2))/(4*nu1*nu2*(1+nu2))
        
        def MD2_22(nu1, nu2):
            if remove_DeltaP_status:
                return MB2_22(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*(3*(3-2*(nu1+nu2))*(1-2*(nu1+nu2)))/(32*nu1*(1+nu1)*nu2*(1+nu2))
        
        def MD3_22(nu1, nu2):
            if remove_DeltaP_status:
                return MB3_22(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2)*(1+2*(nu1**2-4*nu1*nu2+nu2**2)))/(16*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def MD4_22(nu1, nu2):
            if remove_DeltaP_status:
                return MB4_22(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((9-4*(nu1+nu2)**2)*(1-4*(nu1+nu2)**2))/(32*nu1*(1+nu1)*nu2*(1+nu2))
        
        #A function: contributions due to b2 & bs2
        def MtAfkmpfp_22_b2(nu1, nu2):
            return self.Imatrix(nu1,nu2) * ( (2*(nu1+nu2) - 3) * (2*(nu1+nu2) - 1) )/(2*nu1*nu2) 
        
        def MtAfkmpfp_22_bs2(nu1, nu2):
            return self.Imatrix(nu1,nu2) * ( (2*(nu1+nu2) -3) * (2*(nu1+nu2) - 1) * (-1 - nu2 + nu1*(2*nu2 - 1)) )/(6*nu1*(1+nu1)*nu2*(1+nu2))
        
        common_return_values = (
                                M22_dd(nu1, nu2), M22_dt_fp(nu1, nu2), M22_tt_fpfp(nu1, nu2), M22_tt_fkmpfp(nu1, nu2),
                                MtAfp_11(nu1, nu2), MtAfkmpfp_12(nu1, nu2), MtAfkmpfp_22(nu1, nu2), MtAfpfp_22(nu1, nu2), 
                                MtAfkmpfpfp_23(nu1, nu2), MtAfkmpfpfp_33(nu1, nu2), MB1_11(nu1, nu2), MC1_11(nu1, nu2), 
                                MB2_11(nu1, nu2), MC2_11(nu1, nu2), MD2_21(nu1, nu2), MD3_21(nu1, nu2), MD2_22(nu1, nu2), 
                                MD3_22(nu1, nu2), MD4_22(nu1, nu2)
        )
        
        if A_full_status:
            return common_return_values + ( MtAfkmpfp_22_b2(nu1, nu2), MtAfkmpfp_22_bs2(nu1, nu2) )
        else:
            return common_return_values
        
        
    #M22-type Biasing
    def M22bias(self, nu1, nu2):
        
        def MPb1b2(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-4+7*(nu1+nu2)))/(28*nu1*nu2)
        
        def MPb1bs2(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(2+14*nu1**2 *(-1+2*nu2)-nu2*(3+14*nu2)+nu1*(-3+4*nu2*(-11+7*nu2))))/(168*nu1*(1+nu1)*nu2*(1+nu2))
        
        def MPb22(nu1, nu2):
            return 1/2 * self.Imatrix(nu1, nu2)

        def MPb2bs2(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*nu2))/(12*nu1*nu2)

        def MPb2s2(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((63-60*nu2+4*(3*(-5+nu1)*nu1+(17-4*nu1)*nu1*nu2+(3+2*(-2+nu1)*nu1)*nu2**2)))/(36*nu1*(1+nu1)*nu2*(1+nu2))

        def MPb2t(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-4+7*nu1)*(-3+2*(nu1+nu2)))/(14*nu1*nu2)

        def MPbs2t(nu1, nu2):
            return  self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-19-10*nu2+nu1*(39-30*nu2+14*nu1*(-1+2*nu2))))/(84*nu1*(1+nu1)*nu2*(1+nu2))
        
        def MB1_21(nu1, nu2):
            if remove_DeltaP_status:
                return self.Imatrix(nu1, nu2)*(3*(-3 + 2*nu1)*(-3 + 2*nu1 + 2*nu2))/ (8.*nu1*nu2*(1 + nu2)*(-3 + nu1 + nu2))
            else:
                return 0.0 * self.Imatrix(nu1, nu2)
            
        def MB1_22(nu1, nu2):
            if remove_DeltaP_status:
                return self.Imatrix(nu1, nu2)*(-15*(-3 + 2*nu1)*(-3 + 2*nu2)*(-3 + 2*nu1 + 2*nu2)) / (64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
            else:
                return 0.0 * self.Imatrix(nu1, nu2)
        
        #A function: contributions due to b2 & bs2
        def MtAfp_11_b2(nu1, nu2):
            return self.Imatrix(nu1,nu2) * ( 4*(nu1+nu2) - 6)/nu1
        
        def MtAfp_11_bs2(nu1, nu2):
            return self.Imatrix(nu1,nu2) * ( (2*nu1-1) * (2*nu2-3) * (2*(nu1+nu2) - 3) )/(3*nu1*(1+nu1)*nu2)
        
        def MtAfkmpfp_12_b2(nu1, nu2):
            return self.Imatrix(nu1,nu2) * (3 - 2*(nu1+nu2))/(2*nu1*nu2)
        
        def MtAfkmpfp_12_bs2(nu1, nu2):
            return self.Imatrix(nu1,nu2) * ( (5+2*nu1*(nu2-2) - 4*nu2) * (3 - 2*(nu1+nu2)) )/(6*nu1*(1+nu1)*nu2*(1+nu2))
        
        common_return_values = (
                                MPb1b2(nu1, nu2), MPb1bs2(nu1, nu2), MPb22(nu1, nu2), MPb2bs2(nu1, nu2), 
                                MPb2s2(nu1, nu2), MPb2t(nu1, nu2), MPbs2t(nu1, nu2),
                                MB1_21(nu1, nu2), MB1_22(nu1, nu2)
        )
        
        if A_full_status:
            return common_return_values + ( MtAfp_11_b2(nu1, nu2), MtAfp_11_bs2(nu1, nu2), 
                                            MtAfkmpfp_12_b2(nu1, nu2), MtAfkmpfp_12_bs2(nu1, nu2) )
        else:
            return common_return_values
        
        
    #M13-type
    def M13(self, nu1):
        
        #Overdensity and velocity
        def M13_dd(nu1):
            return ((1+9*nu1)/4) * np.tan(nu1*np.pi)/( 28*np.pi*(nu1+1)*nu1*(nu1-1)*(nu1-2)*(nu1-3) )
        
        def M13_dt_fk(nu1):
            return ((-7+9*nu1)*np.tan(nu1*np.pi))/(112*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def M13_tt_fk(nu1):
            return -(np.tan(nu1*np.pi)/(14*np.pi*(-3 + nu1)*(-2 + nu1)*(-1 + nu1)*nu1*(1 + nu1) ))
        
        # A function
        def Mafk_11(nu1):
            return ((15-7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafp_11(nu1):
            return ((-6+7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafkfp_12(nu1):
            return (3*(-13+7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafpfp_12(nu1):
            return (3*(1-7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfkfp_33(nu1):
            return ((21+(53-28*nu1)*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfpfp_33(nu1):
            return ((-21+nu1*(-17+28*nu1))*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        
        return (M13_dd(nu1), M13_dt_fk(nu1), M13_tt_fk(nu1), Mafk_11(nu1),  Mafp_11(nu1), Mafkfp_12(nu1),
                Mafpfp_12(nu1), Mafkfkfp_33(nu1), Mafkfpfp_33(nu1))
    
    
    #M13-type Biasing
    def M13bias(self, nu1):
        
        def Msigma23(nu1):
            return (45*np.tan(nu1*np.pi))/(128*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        return (Msigma23(nu1))
    
        
    #Computation of M22-type matrices            
    def M22type(self, b_nu, M22):
        N = self.nfftlog
        kmin = self.kmin
        kmax = self.kmax
        
        # nuT = -etaT/2, etaT = bias_nu + i*eta_m
        jj = np.arange(N + 1)
        nuT = -0.5 * (b_nu + (2*np.pi*1j/np.log(kmax/kmin)) * (jj - N/2) *(N-1)/(N))

        # reduce time x10 compared to "for" iterations
        nuT_x, nuT_y = np.meshgrid(nuT, nuT)
        M22matrix = M22(nuT_y, nuT_x)

        return np.array(M22matrix)
    
    #Computation of M13-type matrices
    def M13type(self, b_nu, M13):
        N = self.nfftlog
        kmin = self.kmin
        kmax = self.kmax
        
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m
        ii = np.arange(N + 1)
        nuT = -0.5 * (b_nu + (2*np.pi*1j/np.log(kmax/kmin)) * (ii - N/2) *(N-1)/(N))
        M13vector = M13(nuT)

        return np.array(M13vector)
    
    def calculate_matrices(self):
        M22T = self.M22type(b_nu = self.b_nu, M22 = self.M22)
        bnu_b = 15.1 * self.b_nu
        M22biasT = self.M22type(b_nu = bnu_b, M22 = self.M22bias)
        M22matrices = np.concatenate((M22T, M22biasT))
        
        M13T = self.M13type(b_nu = self.b_nu, M13 = self.M13)  
        M13biasT = np.reshape(self.M13type(b_nu = bnu_b, M13 = self.M13bias), (1, self.nfftlog + 1)) 
        M13vectors = np.concatenate((M13T, M13biasT))

        matrices = {
            'M22matrices': M22matrices,
            'M13vectors': M13vectors
        }
        np.save(self.filename, matrices)
        return matrices
    
    def get_mmatrices(self):
        if os.path.exists(self.filename):
            print(f"Loading matrices from {self.filename}")
            matrices = np.load(self.filename, allow_pickle=True).item()
        else:
            print(f"Calculating and saving matrices to {self.filename}")
            matrices = self.calculate_matrices()
        return matrices


# In[7]:


class NonLinearPowerSpectrumCalculator:
    """
    A class to calculate 1-loop corrections to the linear power spectrum.

    Attributes:
        mmatrices (tuple): Set of matrices required for 1-loop computations.
        kernels (str): Choice of kernels ('eds' or 'fk').
        rbao (float): BAO scale in Mpc/h.
        kwargs (dict): Additional optional keyword arguments.

    Notes:
        kminout (float): Minimum k value for the output. This is a fixed value.
        kmaxout (float): Maximum k value for the output. This is a fixed value.
        nk (int): Number of k points for the output. This is a fixed value.
    """
    def __init__(self, mmatrices, kernels='fk', rbao=104., **kwargs):
        self.mmatrices = mmatrices
        self.kernels = kernels
        self.rbao = rbao
        self.kwargs = kwargs
        
        self.kminout = 0.001
        self.kmaxout = 0.5
        self.nk = 120
        self.kTout = np.geomspace(self.kminout, self.kmaxout, num=self.nk)
        
        #FFTLog
        self.kmin = 10**(-7)
        self.kmax = 100.
        self.b_nu = -0.1   # Not yet tested for other values
                
        self.M22matrices = self.mmatrices.get('M22matrices')
        self.M13vectors = self.mmatrices.get('M13vectors')
        self.nfftlog = self.M13vectors.shape[-1] - 1    
        
        
    def _get_f0(self, cosmo=None, k=None):
        """
        Returns f0 from cosmo, kwargs, or computes it if necessary. 
        Raises ValueError if insufficient parameters are provided.
        """
        if 'f0' in self.kwargs:
            return self.kwargs['f0']

        if cosmo is not None and 'z' in self.kwargs:
            return cosmo.scale_independent_growth_factor_f(self.kwargs['z'])
        elif all(p in self.kwargs for p in ['z', 'Omega_m', 'h', 'fnu']):
            if k is None:
                raise ValueError("`k` must be provided to compute f0 from EH fitting function.")
            _, _, f0 = f_over_f0_EH(
                zev=self.kwargs['z'],
                k=k,
                OmM0=self.kwargs['Omega_m'],
                h=self.kwargs['h'],
                fnu=self.kwargs['fnu'],
                Neff=self.kwargs.get('Neff', 3.044),
                Nnu=self.kwargs.get('Nnu', 3)
            )
            return f0
        
    
        else:
            raise ValueError("Insufficient parameters: either provide 'f0' in kwargs, or 'cosmo' and 'z', or 'z', 'Omega_m', 'h', and 'fnu' in kwargs.")
        
    def _initialize_factors(self, cosmo=None, k=None):
        """
        Initializes f(k)/f0 and f0 factors for EdS or fk kernels.
        """
        if self.kernels == 'eds':
            self.inputfkT = None
            self.f0 = self._get_f0(cosmo=cosmo, k=self.inputpkT[0])
            self.Fkoverf0 = np.ones(len(self.kTout), dtype='f8')
        
        else:
            if cosmo is not None and 'z' in self.kwargs:
                self.h = cosmo.h()
                self.fnu = cosmo.Omega_nu/cosmo.Omega0_m()
                self.Omega_m = cosmo.Omega0_m()
                self.inputfkT = f_over_f0_EH(zev=self.kwargs['z'], k=self.inputpkT[0], OmM0=self.Omega_m, h=self.h, fnu=self.fnu, Neff=self.kwargs.get('Neff', 3.044), Nnu=self.kwargs.get('Nnu', 3))
                self.f0 = cosmo.scale_independent_growth_factor_f(self.kwargs['z'])
            elif all(param in self.kwargs for param in ['z', 'Omega_m', 'h', 'fnu']):
                self.inputfkT = f_over_f0_EH(zev=self.kwargs['z'], k=self.inputpkT[0], OmM0=self.kwargs['Omega_m'], h=self.kwargs['h'], fnu=self.kwargs['fnu'], Neff=self.kwargs.get('Neff', 3.044), Nnu=self.kwargs.get('Nnu', 3))
                self.f0 = self.kwargs.get('f0', self.inputfkT[2])
            elif all(param in self.kwargs for param in ['pkttlin', 'f0']):
                inputfkT = list(extrapolate_pklin(k, self.kwargs['pkttlin']))
                inputpkT = list(self.inputpkT)
                fk = (inputfkT[1] / inputpkT[1])**0.5
                self.f0 = self.kwargs.get('f0',fk[0])
                self.inputfkT = (inputfkT[0], fk/self.f0,self.f0)
                
            else:
                raise ValueError("No 'z', 'Omega_m', 'h', 'fnu' provided in kwargs and cosmo is not enabled")
            
            self.Fkoverf0 = interp(self.kTout, self.inputfkT[0], self.inputfkT[1])
            
            
    def _initialize_nonwiggle_power_spectrum(self, inputpkT, pknow=None, cosmo=None,k=None):
        """
        Initializes non-wiggle linear power spectrum.
        """
        if pknow is None:
            if cosmo is not None:
                self.inputpkT_NW = get_pknow(inputpkT[0], inputpkT[1], cosmo.h())
            elif 'h' in self.kwargs:
                self.inputpkT_NW = get_pknow(inputpkT[0], inputpkT[1], self.kwargs['h'])
        else:
            self.inputpkT_NW = extrapolate_pklin(k, pknow)
            
            
            
    def _initialize_liner_power_spectra(self, inputpkT):
        """
        Initializes linear power spectra for density, density-velocity and velocity fields.
        """        
        if self.kernels == 'eds':
            self.inputpkTf = self.inputpkT
            self.inputpkTff = self.inputpkT
            
            self.inputpkTf_NW = self.inputpkT_NW
            self.inputpkTff_NW = self.inputpkT_NW
        else:
            self.inputpkTf = (self.inputpkT[0], self.inputpkT[1] * self.inputfkT[1])
            self.inputpkTff = (self.inputpkT[0], self.inputpkT[1] * self.inputfkT[1]**2)
            
            self.inputpkTf_NW = (self.inputpkT_NW[0], self.inputpkT_NW[1] * self.inputfkT[1])
            self.inputpkTff_NW = (self.inputpkT_NW[0], self.inputpkT_NW[1] * self.inputfkT[1]**2)
            
            
    def _initialize_fftlog_terms(self):
        """
        Initializes fftlog terms: cm coefficients  & etaT 
        """    
        #matter coefficients
        self.cmT = get_cm(self.kmin, self.kmax, self.nfftlog, self.b_nu, self.inputpkT)
        self.cmT_NW = get_cm(self.kmin, self.kmax, self.nfftlog, self.b_nu, self.inputpkT_NW)
        
        #biased tracers coefficients
        self.bnu_b = 15.1 * self.b_nu
        self.cmT_b = get_cm(self.kmin, self.kmax, self.nfftlog, self.bnu_b, self.inputpkT)
        self.cmT_b_NW = get_cm(self.kmin, self.kmax, self.nfftlog, self.bnu_b, self.inputpkT_NW)
        
        if self.kernels == 'eds':
            # Avoid redundant computations
            self.cmTf = self.cmT
            self.cmTff = self.cmT
            self.cmTf_NW = self.cmT_NW
            self.cmTff_NW = self.cmT_NW
        
            self.cmTf_b = self.cmT_b
            self.cmTf_b_NW = self.cmT_b_NW
        else:
            self.cmTf = get_cm(self.kmin, self.kmax, self.nfftlog, self.b_nu, self.inputpkTf)
            self.cmTff = get_cm(self.kmin, self.kmax, self.nfftlog, self.b_nu, self.inputpkTff)
            self.cmTf_NW = get_cm(self.kmin, self.kmax, self.nfftlog, self.b_nu, self.inputpkTf_NW)
            self.cmTff_NW = get_cm(self.kmin, self.kmax, self.nfftlog, self.b_nu, self.inputpkTff_NW)

            self.cmTf_b = get_cm(self.kmin, self.kmax, self.nfftlog, self.bnu_b, self.inputpkTf)
            self.cmTf_b_NW = get_cm(self.kmin, self.kmax, self.nfftlog, self.bnu_b, self.inputpkTf_NW)
        
        #FFTlog: etaT = bias_nu + i*eta_m
        jj = np.arange(self.nfftlog + 1)
        ietam = (2*np.pi*1j/np.log(self.kmax/self.kmin)) * (jj - self.nfftlog/2) *(self.nfftlog-1) / self.nfftlog
        etamT = self.b_nu + ietam
        etamT_b = self.bnu_b + ietam
        self.K = self.kTout
        self.precvec = self.K[:, None]**(etamT)
        self.precvec_b = self.K[:, None]**(etamT_b)
        
        
    def P22type(self, inputpkT, inputpkTf, inputpkTff, cmT, cmTf, cmTff, cmT_b, cmTf_b):
        
        if self.M22matrices is None:
            raise ValueError("M22matrices not provided in mmatrices.")
        
        if A_full_status:
            (M22_dd, M22_dt_fp, M22_tt_fpfp, M22_tt_fkmpfp, 
             MtAfp_11, MtAfkmpfp_12, MtAfkmpfp_22, 
             MtAfpfp_22, MtAfkmpfpfp_23, MtAfkmpfpfp_33, 
             MB1_11, MC1_11, MB2_11, MC2_11, MD2_21, MD3_21, MD2_22, MD3_22, MD4_22, 
             MtAfkmpfp_22_b2, MtAfkmpfp_22_bs2,
             MPb1b2, MPb1bs2, MPb22, MPb2bs2, MPb2s2, MPb2t, MPbs2t,
             MB1_21, MB1_22,
             MtAfp_11_b2, MtAfp_11_bs2, 
             MtAfkmpfp_12_b2, MtAfkmpfp_12_bs2) = self.M22matrices
        else: 
            (M22_dd, M22_dt_fp, M22_tt_fpfp, M22_tt_fkmpfp,
             MtAfp_11, MtAfkmpfp_12, MtAfkmpfp_22,
             MtAfpfp_22, MtAfkmpfpfp_23, MtAfkmpfpfp_33, 
             MB1_11, MC1_11, MB2_11, MC2_11, MD2_21, MD3_21, MD2_22, MD3_22, MD4_22, 
             MPb1b2, MPb1bs2, MPb22, MPb2bs2, MPb2s2, MPb2t, MPbs2t, 
             MB1_21, MB1_22) = self.M22matrices
        
        vec = cmT * self.precvec
        vecf = cmTf * self.precvec
        vecff = cmTff * self.precvec 
        
        vec_b = cmT_b * self.precvec_b
        vecf_b = cmTf_b * self.precvec_b
        
        # Ploop
        P22dd = self.K**3 * np.sum(vec @ M22_dd * vec, axis=-1).real
        P22dt = 2*self.K**3 * np.sum(vecf @ M22_dt_fp * vec, axis=-1).real
        P22tt = self.K**3 * (np.sum(vecff @ M22_tt_fpfp * vec, axis=-1) + np.sum(vecf @ M22_tt_fkmpfp * vecf, axis=-1)).real
        
        # Bias
        Pb1b2 = self.K**3 * np.sum(vec_b @ MPb1b2 * vec_b, axis=-1).real
        Pb1bs2 = self.K**3 * np.sum(vec_b @ MPb1bs2 * vec_b, axis=-1).real
        Pb22 = self.K**3 * np.sum(vec_b @ MPb22 * vec_b, axis=-1).real
        Pb2bs2 = self.K**3 * np.sum(vec_b @ MPb2bs2 * vec_b, axis=-1).real
        Pb2s2 = self.K**3 * np.sum(vec_b @ MPb2s2 * vec_b, axis=-1).real
        Pb2t = self.K**3 * np.sum(vecf_b @ MPb2t * vec_b, axis=-1).real
        Pbs2t = self.K**3 * np.sum(vecf_b @ MPbs2t * vec_b, axis=-1).real

        # A-TNS
        I1udd_1b = self.K**3 * np.sum(vecf @ MtAfp_11 * vec, axis=-1).real
        I2uud_1b = self.K**3 * np.sum(vecf @ MtAfkmpfp_12 * vecf, axis=-1).real
        I3uuu_3b = self.K**3 * np.sum(vecff @ MtAfkmpfpfp_33 * vecf, axis=-1).real
        I2uud_2b = self.K**3 * (np.sum(vecf @ MtAfkmpfp_22 * vecf, axis=-1) + np.sum(vecff @ MtAfpfp_22 * vec, axis=-1)).real
        I3uuu_2b = self.K**3 * np.sum(vecff @ MtAfkmpfpfp_23 * vecf, axis=-1).real

        # D-RSD
        I2uudd_1D = self.K**3 * (np.sum(vecf @ MB1_11 * vecf, axis=-1) + np.sum(vec @ MC1_11 * vecff, axis=-1)).real
        I2uudd_2D = self.K**3 * (np.sum(vecf @ MB2_11 * vecf, axis=-1) + np.sum(vec @ MC2_11 * vecff, axis=-1)).real
        I3uuud_2D = self.K**3 * np.sum(vecf @ MD2_21 * vecff, axis=-1).real
        I3uuud_3D = self.K**3 * np.sum(vecf @ MD3_21 * vecff, axis=-1).real
        I4uuuu_2D = self.K**3 * np.sum(vecff @ MD2_22 * vecff, axis=-1).real
        I4uuuu_3D = self.K**3 * np.sum(vecff @ MD3_22 * vecff, axis=-1).real
        I4uuuu_4D = self.K**3 * np.sum(vecff @ MD4_22 * vecff, axis=-1).real
        
        #new
        I3uuud_1_B = self.K**3 * np.sum(vecf_b @ MB1_21 * vec_b, axis=1).real
        I4uuuu_1_B = self.K**3 * np.sum(vecf_b @ MB1_22 * vec_b, axis=1).real
        
        common_values = (
                        P22dd, P22dt, P22tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, 
                        Pb2t, Pbs2t, I1udd_1b, I2uud_1b, I3uuu_3b, I2uud_2b, 
                        I3uuu_2b, I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D,
                        I4uuuu_2D, I4uuuu_3D, I4uuuu_4D,
                        I3uuud_1_B, I4uuuu_1_B
        )
        
        if A_full_status:
            
            I1udd_1b_b2 = self.K**3 * np.sum(vecf_b @ MtAfp_11_b2 * vec_b, axis=-1).real
            I2uud_1b_b2 = self.K**3 * np.sum(vecf_b @ MtAfkmpfp_12_b2 * vecf_b, axis=-1).real
            I2uud_2b_b2 = self.K**3 * np.sum(vecf @ MtAfkmpfp_22_b2 * vecf, axis=-1).real
            
            I1udd_1b_bs2 = self.K**3 * np.sum(vecf_b @ MtAfp_11_bs2 * vec_b, axis = -1).real
            I2uud_1b_bs2 = self.K**3 * np.sum(vecf_b @ MtAfkmpfp_12_bs2 * vecf_b, axis = -1).real
            I2uud_2b_bs2 = self.K**3 * np.sum(vecf @ MtAfkmpfp_22_bs2 * vecf, axis = -1).real
            
            
            extra_values = (
                            I1udd_1b_b2, I2uud_1b_b2, I2uud_2b_b2,
                            I1udd_1b_bs2, I2uud_1b_bs2, I2uud_2b_bs2
            )
            
            return common_values + extra_values
        
        return common_values
    
    
    def P13type(self, inputpkT, inputpkTf, inputpkTff, inputfkT, cmT, cmTf, cmTff, cmT_b, cmTf_b):
        
        if self.M13vectors is None:
            raise ValueError("M13vectors not provided in mmatrices.")
                        
        (M13_dd, M13_dt_fk, M13_tt_fk, Mafk_11, Mafp_11, Mafkfp_12, Mafpfp_12,
         Mafkfkfp_33, Mafkfpfp_33, Msigma23) = self.M13vectors
        
        sigma2psi = 1/(6 * np.pi**2) * simpson(inputpkT[1], x=inputpkT[0])
        sigma2v = 1/(6 * np.pi**2) * simpson(inputpkTf[1], x=inputpkTf[0])
        sigma2w = 1/(6 * np.pi**2) * simpson(inputpkTff[1], x=inputpkTff[0])
        
        vec = cmT * self.precvec
        vecf = cmTf * self.precvec
        vecff = cmTff * self.precvec
        vecfM13dt_fk = vecf @ M13_dt_fk
        
        vec_b = cmT_b * self.precvec_b
        vecf_b = cmTf_b * self.precvec_b
                
        # Ploop
        P13dd = self.K**3 * (vec @ M13_dd).real - 61/105 * self.K**2 * sigma2psi
        #print('P13dd=', P13dd)
        P13dt = 0.5 * self.K**3 * (self.Fkoverf0[:, None] * vec @ M13_dt_fk + vecfM13dt_fk).real - (23/21*sigma2psi * self.Fkoverf0 + 2/21*sigma2v) * self.K**2
        P13tt = self.K**3 * (self.Fkoverf0 * (self.Fkoverf0[:, None] * vec @ M13_tt_fk + vecfM13dt_fk)).real - (169/105*sigma2psi * self.Fkoverf0 + 4/21 * sigma2v) * self.Fkoverf0 * self.K**2
        
        # Bias
        sigma23 = self.K**3 * (vec_b @ Msigma23).real
        
        # A-TNS
        I1udd_1a = self.K**3 * (self.Fkoverf0[:, None] * vec @ Mafk_11 + vecf @ Mafp_11).real + (92/35*sigma2psi * self.Fkoverf0 - 18/7*sigma2v)*self.K**2
        I2uud_1a = self.K**3 * (self.Fkoverf0[:, None] * vecf @ Mafkfp_12 + vecff @ Mafpfp_12).real - (38/35*self.Fkoverf0 *sigma2v + 2/7*sigma2w)*self.K**2
        I3uuu_3a = self.K**3 * self.Fkoverf0 * (self.Fkoverf0[:, None] * vecf @ Mafkfkfp_33 + vecff @ Mafkfpfp_33).real - (16/35*self.Fkoverf0*sigma2v + 6/7*sigma2w)*self.Fkoverf0*self.K**2
        
        return (P13dd, P13dt, P13tt, sigma23, I1udd_1a, I2uud_1a, I3uuu_3a)
       

    def calculate_P22(self):
        P22 = self.P22type(self.inputpkT, self.inputpkTf, self.inputpkTff, self.cmT, self.cmTf, self.cmTff, self.cmT_b, self.cmTf_b)
        P22_NW = self.P22type(self.inputpkT_NW, self.inputpkTf_NW, self.inputpkTff_NW, self.cmT_NW, self.cmTf_NW, self.cmTff_NW, self.cmT_b_NW, self.cmTf_b_NW)
        return P22, P22_NW
    
    def calculate_P13(self):
        P13overpkl = self.P13type(self.inputpkT, self.inputpkTf, self.inputpkTff, self.inputfkT, self.cmT, self.cmTf, self.cmTff, self.cmT_b, self.cmTf_b)
        P13overpkl_NW = self.P13type(self.inputpkT_NW, self.inputpkTf_NW, self.inputpkTff_NW, self.inputfkT, self.cmT_NW, self.cmTf_NW, self.cmTff_NW, self.cmT_b_NW, self.cmTf_b_NW)
        return P13overpkl, P13overpkl_NW 
    
    
    def calculate_loop_table(self, k, pklin, pknow=None, cosmo=None, **kwargs):
        self.inputpkT = extrapolate_pklin(k, pklin)
        self.kwargs = kwargs
        
        self._initialize_factors(cosmo=cosmo, k=k)
        self._initialize_nonwiggle_power_spectrum(inputpkT=self.inputpkT, pknow=pknow, cosmo=cosmo,k=k)
        self._initialize_liner_power_spectra(inputpkT=self.inputpkT)
        self._initialize_fftlog_terms()
        
        #Computations for Table
        self.pk_l = np.interp(self.kTout, self.inputpkT[0], self.inputpkT[1])
        self.pk_l_NW = np.interp(self.kTout, self.inputpkT_NW[0], self.inputpkT_NW[1])
        
        self.sigma2w = 1 / (6 * np.pi**2) * simpson(self.inputpkTff[1], x=self.inputpkTff[0])
        self.sigma2w_NW = 1 / (6 * np.pi**2) * simpson(self.inputpkTff_NW[1], x=self.inputpkTff_NW[0])
        
        #rbao = 104.
        p = np.geomspace(10**(-6), 0.4, num=100)
        PSL_NW = interp(p, self.inputpkT_NW[0], self.inputpkT_NW[1])
        self.sigma2_NW = 1 / (6 * np.pi**2) * simpson(PSL_NW * (1 - special.spherical_jn(0, p * self.rbao) + 2 * special.spherical_jn(2, p * self.rbao)), x=p)
        self.delta_sigma2_NW = 1 / (2 * np.pi**2) * simpson(PSL_NW * special.spherical_jn(2, p * self.rbao), x=p)
        
        
        P22, P22_NW = self.calculate_P22()
        P13overpkl, P13overpkl_NW = self.calculate_P13()
        
        #print(P13overpkl_NW[0])
        #print(self.kTout)
        #print(P22_NW[0] + P13overpkl_NW[0]*self.pk_l_NW)
        #print(P13overpkl_NW[0])
        #print('=============================================')
        def remove_zerolag(self, k, pk):
            # Originally: interp(10**(-10), kTout, P22_NW[5])
            return pk - extrapolate(k[:2], pk[:2], self.kmin)[1]
        #Below, we use interp() instead of remove_zerolag(), as it gives better results for small values of k
        
        def combine_loop_terms(self, P22, P13overpkl, pk_l, sigma2w, extra_NW=False):
            Ploop_dd = P22[0] + P13overpkl[0]*pk_l
            #print(P13overpkl[0])
            Ploop_dt = P22[1] + P13overpkl[1]*pk_l
            Ploop_tt = P22[2] + P13overpkl[2]*pk_l
            
            Pb1b2 = P22[3]
            Pb1bs2 = P22[4]
            Pb22 = P22[5] - interp(10**(-10), self.kTout, P22[5])   #remove_zerolag(self, self.kTout, P22[5])
            Pb2bs2 = P22[6] - interp(10**(-10), self.kTout, P22[6]) #remove_zerolag(self, self.kTout, P22[6])
            Pb2s2 = P22[7] - interp(10**(-10), self.kTout, P22[7])  #remove_zerolag(self, self.kTout, P22[7])
            sigma23pkl = P13overpkl[3]*pk_l
            Pb2t = P22[8]
            Pbs2t = P22[9]
            
            I1udd_1 = P13overpkl[4]*pk_l + P22[10]
            I2uud_1 = P13overpkl[5]*pk_l + P22[11]
            I2uud_2 = (P13overpkl[6]*pk_l)/self.Fkoverf0 + self.Fkoverf0*P13overpkl[4]*pk_l + P22[13]
            I3uuu_2 = self.Fkoverf0*P13overpkl[5]*pk_l + P22[14]
            I3uuu_3 = P13overpkl[6]*pk_l + P22[12]

            I2uudd_1D = P22[15]; I2uudd_2D = P22[16]; I3uuud_2D = P22[17]
            I3uuud_3D = P22[18]; I4uuuu_2D = P22[19]; I4uuuu_3D = P22[20]
            I4uuuu_4D = P22[21]
            #print('I4uuu_4d', I4uuuu_4D.shape)
            
            #terms below become =0 if when remove_delta=False, i.e. when deltaP is kept.
            I3uuud_1_B = P22[22]  # term f^3*mu^2  I3uuud1D = I3uuud1B + I3uuud1C = 0   
            I4uuuu_1_B = P22[23]  # term f^4*mu^3  I4uuud1D = I4uuud1B + I4uuud1C = 0
            
            common_values = [self.kTout, pk_l, self.Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt,
                             Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, sigma23pkl, Pb2t, Pbs2t,
                             I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, I2uudd_1D, 
                             I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D,
                             I4uuuu_4D,
                             I3uuud_1_B, I4uuuu_1_B, 
                             #, sigma2w #, self.f0
                            ]
            
            if A_full_status:
                #A function: b2 and bs2 contributions
                I1udd_1_b2 = P22[22];    I1udd_1_bs2 = P22[25];       
                I2uud_1_b2 = P22[23];    I2uud_1_bs2 = P22[26];    
                I2uud_2_b2 = P22[24];    I2uud_2_bs2 = P22[27];    
                
                common_values.extend([I1udd_1_b2, I2uud_1_b2, I2uud_2_b2,
                                      I1udd_1_bs2, I2uud_1_bs2, I2uud_2_bs2
                                    ])
                
            common_values.append(sigma2w)
            
            if extra_NW:
                common_values.extend([self.sigma2_NW, self.delta_sigma2_NW])
            
            common_values.append(self.f0)
            
            return tuple(common_values)
        
        self.TableOut = combine_loop_terms(self, P22, P13overpkl, self.pk_l, self.sigma2w)
        self.TableOut_NW = combine_loop_terms(self, P22_NW, P13overpkl_NW, self.pk_l_NW, self.sigma2w_NW, extra_NW=True)
        
        return (self.TableOut, self.TableOut_NW)


# In[8]:


class RSDMultipolesPowerSpectrumCalculator:
    """
    A class to calculate the redshift space power spectrum multipoles with flexible bias schemes.
    
    """
    def __init__(self, model="EFT"):
        """
        Initializes the calculator with fixed configuration.

        Args:
            model (str): Model name ('EFT', 'TNS', 'FOLPSD').
        """
        self.model = model
        
    def set_bias_scheme(self, pars, bias_scheme="folps"):
        """Sets the nuisance parameters based on the selected bias scheme."""
        if bias_scheme in ["folps", "pat", "mcdonald"]:
            if pars is None:
                pars = [1.0, 0.5, 0.3, 0.1, 0.01, 0.02, 0.03, 0.04, 0.001, 0.002, 0.003, 0.0]
            (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p) = pars
        
        elif bias_scheme in ["Assassi", "classpt"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for Assassi/classpt bias scheme.")
            (b1_classPT, b2_classPT, bG2_classPT, bGamma3_classPT, alpha0, alpha2, alpha4, 
             ctilde, alphashot0, alphashot2, PshotP, X_FoG_p) = pars
            b1 = b1_classPT
            b2 = b2_classPT - 4/3 * bG2_classPT
            bs2 = 2 * bG2_classPT
            b3nl = -32/21 * (bG2_classPT + 2/5 * bGamma3_classPT)
            
            pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                    ctilde, alphashot0, alphashot2, PshotP, X_FoG_p]
        
        else:
            raise ValueError("Invalid bias scheme. Choose from 'folps', 'pat', 'mcdonald', 'Assassi', or 'classpt'.")
        
        return pars
     
    def interp_table(self, k, table, A_full_status):
        """Interpolation of non-linear terms given by the power spectra."""
        def interp(k, x, y):  # out-of-range below
            from scipy.interpolate import CubicSpline
            return CubicSpline(x, y)(k)
    
        extra = 6 if A_full_status else 0
        return tuple(np.moveaxis(interp(k, table[0], np.column_stack(table[1:28+extra])), -1, 0)) + table[28+extra:]

    def k_ap(self, kobs, muobs, qper, qpar):
        """Return the true wave-number ‘k_AP’."""
        F = qpar / qper
        return (kobs / qper) * (1 + muobs**2 * (1. / F**2 - 1))**0.5

    def mu_ap(self, muobs, qper, qpar):
        """Return the true ‘mu_AP’."""
        F = qpar / qper
        return (muobs / F) * (1 + muobs**2 * (1 / F**2 - 1))**-0.5

    def get_eft_pkmu(self, kev, mu, pars, table, damping='lor'):
        """Calculate the EFT galaxy power spectrum in redshift space."""
        (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p) = pars
        
        #modificar esta cosa, esta mal puesta
        Winfty_all=True       #change to False for VDG and no analytical marginalization
        
        
        if A_full_status:
            (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
             Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
             I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D,
             I3uuud_1B, I4uuuu_1B,
             I1udd_1_b2, I2uud_1_b2, I2uud_2_b2, I1udd_1_bs2, I2uud_1_bs2, I2uud_2_bs2,
             sigma2w, *_, f0) = table
        else:
            (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
             Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
             I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, 
             I3uuud_1B, I4uuuu_1B, 
             sigma2w, *_, f0) = table
            
        #print(I3uuud_1B.shape)
        #print(I3uuud_1B)
        #print("==")
        #print(I3uuud_2D.shape)
        
        fk = Fkoverf0 * f0
        Pdt_L = pkl * Fkoverf0
        Ptt_L = pkl * Fkoverf0**2

        def PddXloop(b1, b2, bs2, b3nl):
            return (b1**2 * Ploop_dd + 2 * b1 * b2 * Pb1b2 + 2 * b1 * bs2 * Pb1bs2 + b2**2 * Pb22
                    + 2 * b2 * bs2 * Pb2bs2 + bs2**2 * Pb2s2 + 2 * b1 * b3nl * sigma23pkl)

        def PdtXloop(b1, b2, bs2, b3nl):
            return b1 * Ploop_dt + b2 * Pb2t + bs2 * Pbs2t + b3nl * Fkoverf0 * sigma23pkl

        def PttXloop(b1, b2, bs2, b3nl):
            return Ploop_tt

        def Af(mu, f0):
            return (f0 * mu**2 * I1udd_1 + f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
                    + f0**3 * (mu**4 * I3uuu_2 + mu**6 * I3uuu_3))
        
        def Af_b2(mu, f0):
            return (f0*mu**2 * I1udd_1_b2 +  f0**2 * (mu**2 * I2uud_1_b2 +  mu**4 * I2uud_2_b2) )
    
        def Af_bs2(mu, f0):
            return (f0*mu**2 * I1udd_1_bs2 +  f0**2 * (mu**2 * I2uud_1_bs2 +  mu**4 * I2uud_2_bs2) )

        def Df(mu, f0):
            return (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D)
                    + f0**3 * (mu**2 * I3uuud_1B + mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
                    + f0**4 * (mu**2 * I4uuuu_1B + mu**4 * I4uuuu_2D + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D))

        def ATNS(mu, b1):
            return b1**3 * Af(mu, f0 / b1)
        
        def ATNS_b2_bs2(mu, b1, b2, bs2):
            return b1**3 * Af_b2(mu, f0/b1) * b2/(2*b1) +  b1**3 * Af_bs2(mu, f0/b1) * bs2/(2*b1)

        def DRSD(mu, b1):
            return b1**4 * Df(mu, f0 / b1)

        def GTNS(mu, b1):
            if remove_DeltaP_status:
                return 0
            else:
                return -((kev * mu * f0)**2 * sigma2w * (b1**2 * pkl + 2 * b1 * f0 * mu**2 * Pdt_L + f0**2 * mu**4 * Ptt_L))

        def PloopSPTs(mu, b1, b2, bs2, b3nl):
        
            if A_full_status:
                return (
                        PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                        + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                        + GTNS(mu, b1) + ATNS_b2_bs2(mu, b1, b2, bs2)
                )
        
            else: 
                return (
                    PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                    + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                    + GTNS(mu, b1)
                )

        def PKaiserLs(mu, b1):
            return (b1 + mu**2 * fk)**2 * pkl

        def PctNLOs(mu, b1, ctilde):
            return ctilde * (mu * kev * f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)

        def Pcts(mu, alpha0, alpha2, alpha4):
            return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4) * kev**2 * pkl

        def Pshot(mu, alphashot0, alphashot2, PshotP):
            return PshotP * (alphashot0 + alphashot2 * (kev * mu)**2)
        
        def Winfty(mu, X_FoG_p):
            lambda2= (f0*kev*mu*X_FoG_p)**2
            exp = - lambda2 * sigma2w /(1+lambda2)
            W   =np.exp(exp) / np.sqrt(1+lambda2)
            return W 

        def Wexp(mu, X_FoG_p):
            lambda2= (f0*kev*mu*X_FoG_p)**2
            exp = - lambda2 * sigma2w
            W   =np.exp(exp)
            return W  

        def Wlorentz(mu, X_FoG_p):
            lambda2= (f0*kev*mu*X_FoG_p)**2
            x2 = lambda2 * sigma2w
            W   = 1.0/(1.0+x2)
            return W 
        
        if damping==None: 
            W=1  # EFT if keeps DeltaP
        elif damping=='exp':
            W=Wexp(mu, X_FoG_p)
        elif damping=='lor':
            W=Wlorentz(mu, X_FoG_p)
        elif damping=='vdg':
            W=Winfty(mu, X_FoG_p)
        
        PK = W*PloopSPTs(mu, b1, b2, bs2, b3nl)  + Pshot(mu, alphashot0, alphashot2, PshotP)

        if Winfty_all==False:
            W = 1.0
                
        return PK + W * ( Pcts(mu, alpha0, alpha2, alpha4) + PctNLOs(mu, b1, ctilde) )

    def get_rsd_pkmu(self, k, mu, pars, table, table_now, IR_resummation=True, damping='lor'):
        """Return redshift space P(k, mu) given input tables."""
        table = self.interp_table(k, table, A_full_status)
        table_now = self.interp_table(k, table_now, A_full_status)
        b1 = pars[0]
        f0 = table[-1]
        fk = table[1] * f0
        pkl, pkl_now = table[0], table_now[0]
        sigma2, delta_sigma2 = table_now[-3:-1]
        # Sigma² tot for IR-resummations, see eq.~ 3.59 at arXiv:2208.02791
        if IR_resummation:
            sigma2t = (1 + f0*mu**2 * (2 + f0))*sigma2 + (f0*mu)**2 * (mu**2 - 1) * delta_sigma2
        else:
            sigma2t =0
        pkmu = ((b1 + fk * mu**2)**2 * (pkl_now + np.exp(-k**2 * sigma2t)*(pkl - pkl_now)*(1 + k**2 * sigma2t))
                 + np.exp(-k**2 * sigma2t) * self.get_eft_pkmu(k, mu, pars, table, damping)
                 + (1 - np.exp(-k**2 * sigma2t)) * self.get_eft_pkmu(k, mu, pars, table_now, damping))
        return pkmu

    def get_rsd_pkell(self, kobs, qpar, qper, pars, table, table_now,
                      bias_scheme="folps", damping='lor', nmu=6, ells=(0, 2, 4), IR_resummation=True):
        """
        Computes the redshift-space power spectrum multipoles P_ell(k).

        Args:
            kobs (array): Observed k.
            qpar (float): Parallel AP parameter.
            qper (float): Perpendicular AP parameter.
            pars (list): Nuisance parameters.
            table (list): table.
            table_now (list): No-wiggle table.
            bias_scheme (str): Bias scheme to use.
            nmu (int): Number of points for GL integration
            ells (tuple): Multipoles
            IR_resummation (bool): Whether to apply IR resummation.

        Returns:
            array: Power spectrum multipoles for each ell.
        """
        pars = self.set_bias_scheme(pars, bias_scheme=bias_scheme)
        
        def weights_leggauss(nx, sym=False):
            """Return weights for Gauss-Legendre integration."""
            import numpy as np
            x, wx = np.polynomial.legendre.leggauss((1 + sym) * nx)
            if sym:
                x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
            return x, wx

        muobs, wmu = weights_leggauss(nmu, sym=True)
        wmu = np.array([wmu * (2 * ell + 1) * legendre(ell)(muobs) for ell in ells])
        jac, kap, muap = (qpar * qper**2)**(-3), self.k_ap(kobs[:, None], muobs, qpar, qper), self.mu_ap(muobs, qpar, qper)[None, :]
        #print(muap[0])
        pkmu = jac * self.get_rsd_pkmu(kap, muap, pars, table, table_now, IR_resummation, damping)
        return np.sum(pkmu * wmu[:, None, :], axis=-1)     


# In[9]:


class BispectrumCalculator:
    def __init__(self, basis='sugiyama'):
        """
        basis : str
            'sugiyama' or 'scoccimarro' (currently only 'sugiyama' is implemented)
        """
        self.basis = basis.lower()
        if self.basis not in ['sugiyama', 'scoccimarro']:
            raise ValueError("basis must be 'sugiyama' or 'scoccimarro'.")
            
            
    def pklIR_f(self, k,pklIRT):
        return np.interp(k, pklIRT[0], pklIRT[1])

            
    #GL pairs [[x1,w1],[x2,w2],....
    def tablesGL_f(self, precision=[4, 5, 5]):
        Nphi, Nx, Nmu = precision
        Pi = np.pi

        phi_roots, phi_weights = scipy.special.roots_legendre(Nphi)
        phi_roots = Pi / 2 * phi_roots + Pi / 2
        phi_weights = Pi / 2 * phi_weights
        phiGL = np.array([phi_roots, phi_weights]).T

        x_roots, x_weights = scipy.special.roots_legendre(Nx)
        xGL = np.array([x_roots, x_weights]).T

        mu_roots, mu_weights = scipy.special.roots_legendre(Nmu)
        muGL = np.array([mu_roots, mu_weights]).T

        return [phiGL, xGL, muGL]

    def kAP(self, k, mu, qpar, qperp):
        return k / qperp * np.sqrt(1 + mu**2 * (-1 + (qperp**2) / (qpar**2)))

    def muAP(self, mu, qpar, qperp):
        return (mu * qperp / qpar) / np.sqrt(1 + mu**2 * (-1 + (qperp**2) / (qpar**2)))

    def APtransforms(self, k1, k2, x12, mu1, cosphi, qpar, qperp):
        k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
        mu2 = np.sqrt(1 - mu1**2) * np.sqrt(1 - x12**2) * cosphi + mu1 * x12
        mu3 = -k1 / k3 * mu1 - k2 / k3 * mu2

        k1AP = self.kAP(k1, mu1, qpar, qperp)
        k2AP = self.kAP(k2, mu2, qpar, qperp)
        k3AP = self.kAP(k3, mu3, qpar, qperp)

        mu1AP = self.muAP(mu1, qpar, qperp)
        mu2AP = self.muAP(mu2, qpar, qperp)
        mu3AP = self.muAP(mu3, qpar, qperp)

        x12AP = (k3AP**2 - k1AP**2 - k2AP**2) / (2 * k1AP * k2AP)
        x31AP = -(k1AP + k2AP * x12AP) / k3AP
        x23AP = -(k2AP + k1AP * x12AP) / k3AP

        return np.array([k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP, cosphi])
            

    def Qij(self, ki, kj, xij, mui, muj, f, bpars):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = bpars

        fi = f
        fj = f
        fij = f

        Z1i = b1 + fi * mui**2
        Z1j = b1 + fj * muj**2
        # Z1efti = Z1i - c1*(ki*mui)**2
        # Z1eftj = Z1j - c1*(kj*muj)**2 
        
        kmu = ki * mui + kj * muj
        mu2 = kmu**2 / (ki**2 + kj**2 + 2 * ki * kj * xij)
        crossterm = 0.5 * kmu * (fj * muj / kj * Z1i + fi * mui / ki * Z1j)

        advection = xij / 2.0 * (ki / kj + kj / ki)
        F2 = 5.0 / 7.0 + 2.0 / 7.0 * xij**2 + advection
        G2 = 3.0 / 7.0 + 4.0 / 7.0 * xij**2 + advection

        Z2 = b1 * F2 + fij * mu2 * G2 + crossterm + b2 / 2.0 + bs * (xij**2 - 1.0 / 3.0)
        Qij = 2 * Z2
        return Qij

    def bispectrum(self, k1, k2, x12, mu1, phi, f, f0, sigma2v, bpars, qpar, qperp, pk_in, damping = 'lor'):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = bpars

        cosphi = np.cos(phi)
        APtransf = self.APtransforms(k1, k2, x12, mu1, cosphi, qpar, qperp)
        k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP, cosphi = APtransf

        Q12 = self.Qij(k1AP, k2AP, x12AP, mu1AP, mu2AP, f, bpars)
        Q13 = self.Qij(k1AP, k3AP, x31AP, mu1AP, mu3AP, f, bpars)
        Q23 = self.Qij(k2AP, k3AP, x23AP, mu2AP, mu3AP, f, bpars)

        pk1 = self.pklIR_f(k1AP, pk_in)
        pk2 = self.pklIR_f(k2AP, pk_in)
        pk3 = self.pklIR_f(k3AP, pk_in)

        f1 = f2 = f3 = f
        Z1_1 = b1 + f1 * mu1AP**2
        Z1_2 = b1 + f2 * mu2AP**2
        Z1_3 = b1 + f3 * mu3AP**2

        Z1eft1 = Z1_1 - (c1 * mu1AP**2 + c2 * mu1AP**4) * k1AP**2
        Z1eft2 = Z1_2 - (c1 * mu2AP**2 + c2 * mu2AP**4) * k2AP**2
        Z1eft3 = Z1_3 - (c1 * mu3AP**2 + c2 * mu3AP**4) * k3AP**2

        B12 = Q12 * Z1eft1 * pk1 * Z1eft2 * pk2
        B13 = Q13 * Z1eft1 * pk1 * Z1eft3 * pk3
        B23 = Q23 * Z1eft3 * pk2 * Z1eft3 * pk3

        l2 = (k1AP * mu1AP)**2 + (k2AP * mu2AP)**2 + (k3AP * mu3AP)**2
        l2 = 0.5 * l2 * (f0 * X_FoG_b)**2

        Winfty = np.exp(- l2 * sigma2v / (1 + l2)) / np.sqrt((1 + l2)**3)
        Wlor = 1.0 / (1.0 + l2 * sigma2v)

        #damping = 'lor'
        if damping is None:
            W = 1
        elif damping == 'lor':
            W = Wlor
        elif damping == 'vdg':
            W = Winfty
        
        ## Noise 
        # To match eq.3.14 of 2110.10161, one makes (1+Pshot) -> (1+Pshot)/bar-n; Bshot -> Bshot/bar-n
        shot = (b1 * Bshot + 2.0 * (1 + Pshot) * f1 * mu1AP**2) * Z1eft1 * pk1 \
             + (b1 * Bshot + 2.0 * (1 + Pshot) * f2 * mu2AP**2) * Z1eft2 * pk2 \
             + (b1 * Bshot + 2.0 * (1 + Pshot) * f3 * mu3AP**2) * Z1eft3 * pk3 \
             + (1 + Pshot)**2

        bispectrum = W * (B12 + B13 + B23) + shot
        alpha = qpar * qperp**2
        bispectrum = bispectrum / alpha**2

        return bispectrum

    def Bisp_Sugiyama(self, f, f0, bpars, pk_input, z_pk,
                      k1k2pairs, qpar, qperp, precision=[4, 5, 5], damping = 'lor'):

        #OmM, h = bisp_cosmo_params
        #qperp, qpar = 1, 1

        #if Omfid > 0:
        #    qperp = DA(OmM, z_pk) / DA(Omfid, z_pk)
        #    qpar = Hubble(Omfid, z_pk) / Hubble(OmM, z_pk)

        #f = f0_function(z_pk, OmM)

        sigma2v_ = integrate.simps(pk_input[1], pk_input[0]) / (6 * np.pi**2)
        sigma2v_ *= 1.05 #correction due to k cut
        
        # tables for GL pairs [phi,mu,x] [[x1,w1],[x2,w2],....]
        tablesGL = self.tablesGL_f(precision)
        size = len(k1k2pairs)

        B000 = np.zeros(size)
        B202 = np.zeros(size)

        for ii in range(size):
            k1, k2 = k1k2pairs[ii]
            B000[ii], B202[ii] = self.Sugiyama_B000_B202(
                k1, k2, f, f0, sigma2v_, bpars, qpar, qperp, tablesGL, pk_input, damping)

        return B000, B202

    def Sugiyama_B000_B202(self, k1, k2, f, f0, sigma2v, bpars, qpar, qperp, tablesGL, pk_in, damping = 'lor'):
        phiGL, xGL, muGL = tablesGL

        phi_values, phi_weights = phiGL[:, 0], phiGL[:, 1]
        mu_values, mu_weights = muGL[:, 0], muGL[:, 1]
        x_values, x_weights = xGL[:, 0], xGL[:, 1]

        fourpi = 4 * np.pi
        normB000 = 0.5 / fourpi
        normB202 = 5.0 / 2.0 / fourpi

        x_mesh, mu_mesh, phi_mesh = np.meshgrid(x_values, mu_values, phi_values, indexing='ij')

        bisp = self.bispectrum(k1, k2, x_mesh, mu_mesh, phi_mesh,
                               f, f0, sigma2v, bpars, qpar, qperp, pk_in, damping)

        int_phi = 2 * np.sum(bisp * phi_weights, axis=2)
        int_mu_B000 = np.sum(int_phi * mu_weights, axis=1)
        int_all_B000 = np.sum(int_mu_B000 * x_weights)

        leg2 = 0.5 * (-1.0 + 3.0 * mu_values**2)
        int_mu_B202 = np.sum(int_phi * leg2 * mu_weights, axis=1)
        int_all_B202 = np.sum(int_mu_B202 * x_weights)

        B000 = int_all_B000 * normB000
        B202 = int_all_B202 * normB202

        return B000, B202


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




