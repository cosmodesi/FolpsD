# ============================================================================================ #
#                                             FOLPS                                            #
# ============================================================================================ #
#     Fast and Efficient Computation of the Redshift-Space Power Spectrum and Bispectrum       #
#     in LCDM and Cosmological Models with Massive Neutrinos and Modified Gravity Theories     #
# ============================================================================================ #

import os
import scipy
from scipy import special
from scipy import integrate
from scipy.interpolate import interp1d

from scipy.interpolate import RegularGridInterpolator
# from triumvirate._arrayops import reshape_threept_datatab

try:
    from triumvirate.winconv import (
        WinConvFormulae,
    )
except ImportError:
    WinConvFormulae = None



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
                # import jax  # type: ignore
                # if any(device.device_kind in ["Gpu", "Metal"] for device in jax.devices()):
                #     print("✅ GPU detected. Using JAX with GPU.")
                # else:
                #     print("⚠️ No GPU found. Using JAX with CPU.")
                import jax
                devices = jax.devices()
                backend = jax.default_backend()
                if backend == "gpu":
                    print(f"✅ GPU detected. Using JAX with GPU. Devices: {devices}")
                elif backend == "cpu":
                    print(f"⚠️ JAX is using CPU. Devices: {devices}")
                else:
                    print(f"ℹ️ JAX backend: {backend}. Devices: {devices}")

                from jax import numpy as np
                try:
                    from folps.tools_jax import (
                        interp,
                        interp2d,
                        simpson,
                        legendre,
                        extrapolate,
                        extrapolate_pklin,
                        get_pknow,
                        get_linear_ir,
                        get_linear_ir_ini,
                        qpar_qperp,
                        get_pknow_jax,
                        interp_at_kmin,
                        cubic_spline_not_a_knot_eval,
                        reshape_threept_datatab_jax
                    )
                except ModuleNotFoundError:
                    # Support running scripts from inside the folps/ directory
                    # where this file is imported as a module, not as a package.
                    from tools_jax import (
                        interp,
                        interp2d,
                        simpson,
                        legendre,
                        extrapolate,
                        extrapolate_pklin,
                        get_pknow,
                        get_linear_ir,
                        get_linear_ir_ini,
                        qpar_qperp,
                        get_pknow_jax,
                        interp_at_kmin,
                        cubic_spline_not_a_knot_eval,
                        reshape_threept_datatab_jax
                    )
                self.modules = {
                    "np": np,
                    "interp": interp,
                    "interp2d": interp2d,
                    "simpson": simpson,
                    "legendre": legendre,
                    "extrapolate": extrapolate,
                    "extrapolate_pklin": extrapolate_pklin,
                    "get_pknow": get_pknow,
                    "get_linear_ir":get_linear_ir,
                    "get_linear_ir_ini":get_linear_ir_ini,
                    "qpar_qperp": qpar_qperp,
                    "get_pknow_jax": get_pknow_jax,
                    "interp_at_kmin": interp_at_kmin,
                    "cubic_spline_eval": cubic_spline_not_a_knot_eval,
                    "reshape_threept_datatab_jax": reshape_threept_datatab_jax,
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
            try:
                from folps.tools import interp, simpson, legendre, extrapolate, extrapolate_pklin, get_pknow, get_linear_ir, get_linear_ir_ini, qpar_qperp
            except ModuleNotFoundError:
                # Support running scripts from inside the folps/ directory
                # where this file is imported as a module, not as a package.
                from tools import interp, simpson, legendre, extrapolate, extrapolate_pklin, get_pknow, get_linear_ir, get_linear_ir_ini, qpar_qperp
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
                "qpar_qperp": qpar_qperp,
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
qpar_qperp = backend_manager.get_module("qpar_qperp")
backend = backend_manager.backend

# `get_pknow_jax` and `interp_at_kmin` exist only when JAX backend is selected.
# Define them conditionally so code that imports `folps` can safely reference the name
# regardless of backend.
if backend == 'jax':
    # try to fetch the JAX-specific implementation; fall back to None if absent
    get_pknow_jax = backend_manager.get_module("get_pknow_jax")
    interp_at_kmin = backend_manager.get_module("interp_at_kmin")
    cubic_spline_eval = backend_manager.get_module("cubic_spline_eval")
    interp2d = backend_manager.get_module("interp2d")
    reshape_threept_datatab_jax = backend_manager.get_module("reshape_threept_datatab_jax")
else:
    get_pknow_jax = None
    interp_at_kmin = None
    cubic_spline_eval = None
    interp2d=None

#using_jax = backend_manager.using_jax


def spherical_jn_backend(n, x):
    """Backend-aware spherical Bessel function j_n(x) for small set of orders.

    This function uses the currently-selected `np` (either numpy or jax.numpy)
    so it works with JAX tracers inside jax.jit. Only j0 and j2 are required in
    the codebase; for other n it will try to fall back to scipy on CPU (which
    will fail under jax tracers) and raise a clear error.
    """
    x = np.asarray(x)
    # j0(x) = sin(x)/x with limit 1 at x->0
    if n == 0:
        # Avoid evaluating divisions at x=0 in NumPy while keeping JAX compatibility.
        x_safe = np.where(x == 0, np.array(1.0, dtype=x.dtype), x)
        return np.where(x == 0, np.array(1.0, dtype=x.dtype), np.sin(x) / x_safe)

    # j2(x) = (3/x^2 - 1) * sin(x)/x - 3*cos(x)/x^2
    if n == 2:
        sinx = np.sin(x)
        cosx = np.cos(x)
        x2 = x * x
        # avoid division by zero using a small-x series expansion
        small = np.abs(x) < 1e-6
        x_safe = np.where(x == 0, np.array(1.0, dtype=x.dtype), x)
        x2_safe = np.where(x2 == 0, np.array(1.0, dtype=x2.dtype), x2)
        sinx_over_x = sinx / x_safe
        j2 = ((3.0 / x2_safe - 1.0) * sinx_over_x) - (3.0 * cosx / x2_safe)
        # series limit j2 ~ x^2/15 for x->0
        j2_safe = np.where(small, x2 / 15.0, j2)
        return j2_safe

    # fallback: try scipy for generic n when not running under JAX
    try:
        from scipy import special as _special
        # Convert to host numpy array for scipy, then back to backend np
        x_host = np.asarray(x)
        # If using jax, x may be a tracer — attempting to call scipy will fail.
        return _special.spherical_jn(n, x_host)
    except Exception:
        raise NotImplementedError(f"spherical_jn for n={n} is not implemented for the current backend")



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

    # Choose odeint depending on backend
    # For numpy/scipy use scipy.integrate.odeint(func, y0, t)
    # For jax use jax.experimental.ode.odeint(func, y0, t)
    use_jax = (backend == 'jax')

    if use_jax:
        from jax.experimental.ode import odeint as jax_odeint
    else:
        from scipy.integrate import odeint as scipy_odeint

    # differential eq. -- make sure it returns the same array-like type for both odeint flavors
    def Deqs(y, eta_val):
        # y is [D, Dprime]
        D = y[0]
        Dprime = y[1]
        return np.array([Dprime, f2(eta_val) * D - f1(eta_val) * Dprime])

    #eta range and initial conditions
    eta_arr = np.linspace(etaini, etafin, 1001)
    Df0 = np.exp(etaini)
    Df_p0 = np.exp(etaini)
    y0 = np.array([Df0, Df_p0])

    if use_jax:
        # jax.experimental.ode.odeint expects signature func(y, t, *args)
        # jax returns an array with shape (len(t), len(y))
        sol = jax_odeint(Deqs, y0, eta_arr)
        # sol is array [[D, Dprime], ...]
        Dplus = sol[:, 0]
        Dplusp = sol[:, 1]
    else:
        # scipy.integrate.odeint returns array with shape (len(t), len(y))
        sol = scipy_odeint(lambda yy, tt: Deqs(yy, tt), y0, eta_arr)
        Dplus = sol[:, 0]
        Dplusp = sol[:, 1]

    # interpolate to requested redshift
    Dplusp_ = interp(etaofz(zev), eta_arr, Dplusp)
    Dplus_ = interp(etaofz(zev), eta_arr, Dplus)
    f0 = Dplusp_ / Dplus_

    return (k, fFit, f0)


def f_over_f0_EH_jax_optimized(zev, k, OmM0, h, fnu, Nnu=3, Neff=3.044):
    """
    JAX-optimized version of f_over_f0_EH for high-performance computation.

    This function provides the same computation as f_over_f0_EH but is specifically
    optimized for JAX with JIT compilation, resulting in significant speedup
    (typically 500x+ faster when JIT-compiled).

    Args:
        zev: redshift
        k: wave-number
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        h = H0/100
        fnu: Omega_nu/OmM0
        Nnu: number of massive neutrinos
        Neff: effective number of neutrinos
    Returns:
        (k, fFit, f0) - same as original f_over_f0_EH
    """
    # Only available when using JAX backend
    if backend != 'jax':
        raise RuntimeError("f_over_f0_EH_jax_optimized requires JAX backend. "
                         "Use f_over_f0_EH for NumPy backend.")

    # Import JAX numpy for this function
    import jax.numpy as jnp

    eta = jnp.log(1 / (1 + zev))   # log of scale factor
    omrv = 2.469e-5/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff))  # rad: including neutrinos
    aeq = omrv/OmM0           # matter-radiation equality

    pcb = 5./4 - jnp.sqrt(1 + 24*(1 - fnu))/4     # neutrino suppression
    c = 0.7
    theta272 = (1.00)**2                          # T_{CMB} = 2.7*(theta272)
    pf = (k * theta272)/(OmM0 * h**2)
    DEdS = jnp.exp(eta)/aeq                       # growth function: EdS cosmology

    fnunonzero = jnp.where(fnu != 0., fnu, 1.)
    yFS = 17.2*fnu*(1 + 0.488*fnunonzero**(-7/6))*(pf*Nnu/fnunonzero)**2  # yFreeStreaming
    # pcb = 0. and yFS = 0. when fnu = 0.
    rf = DEdS/(1 + yFS)
    fFit = 1 - pcb/(1 + (rf)**c)                  # f(k)/f0

    # Getting f0 - optimized JAX implementation
    def OmM(eta):
        return 1./(1. + ((1-OmM0)/OmM0)*jnp.exp(3*eta))

    def f1(eta):
        return 2. - 3./2. * OmM(eta)

    def f2(eta):
        return 3./2. * OmM(eta)

    etaini = -6  # initial eta, early enough to evolve as EdS (D ∝ a)
    zfin = -0.99

    def etaofz(z):
        return jnp.log(1/(1 + z))

    etafin = etaofz(zfin)

    from jax.experimental.ode import odeint

    # differential eq.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return jnp.array([Dprime, f2(eta)*Df - f1(eta)*Dprime])

    # eta range and initial conditions
    eta_range = jnp.linspace(etaini, etafin, 1001)
    Df0 = jnp.exp(etaini)
    Df_p0 = jnp.exp(etaini)

    # solution
    solution = odeint(Deqs, jnp.array([Df0, Df_p0]), eta_range)
    Dplus = solution[:, 0]
    Dplusp = solution[:, 1]

    # Interpolate to requested redshift using JAX interp
    eta_target = etaofz(zev)
    Dplusp_ = jnp.interp(eta_target, eta_range, Dplusp)
    Dplus_ = jnp.interp(eta_target, eta_range, Dplus)
    f0 = Dplusp_/Dplus_

    return (k, fFit, f0)


# JIT-compiled version for maximum performance
if backend == 'jax':
    import jax
    f_over_f0_EH_jax_jit = jax.jit(f_over_f0_EH_jax_optimized)
else:
    f_over_f0_EH_jax_jit = None

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
        nfftlog (int, optional)       : Number of sample points for FFTLog integration. Defaults to 128.
                                        It is recommended to use this default value for numerical accuracy;
                                        see Figure 8 in arXiv:2208.02791.
        A_full (bool, optional)       : Whether to compute the full A_TNS function. If True (default), the
                                        function includes contributions from b1, b2, and bs2. If False, it
                                        uses an approximation based only on the linear bias b1.
        use_TNS_model (bool, optional): use_TNS_model=True computes P_lin + P_loop - Delta P(k,mu)   (using the notation of 2501.18597)
                                        # Default: use_TNS_model=False computes P_lin + P_loop
        save_dir (str, optional)        : Directory to save (or retrieve) the computed M matrices.
                                        #If None, matrices will not be saved to disk. Defaults to None.


    Notes:
        - The wavenumber range (k) is fixed internally to kmin = 1e-7 and kmax = 100.
        - The FFT bias parameter b_nu is fixed to -0.1. Other values have not been tested.

    Returns:
        list: A list containing all computed M matrices.
    """
    def __init__(self, nfftlog=128, A_full=True, use_TNS_model=False,save_dir=None):
        global A_full_status, use_TNS_model_status
        self.nfftlog = nfftlog
        self.kmin = 10**(-7)
        self.kmax = kmax=100.
        self.b_nu = -0.1  # not yet tested for other values
        self.A_full = A_full
        A_full_status = A_full


        self.use_TNS_model = use_TNS_model
        use_TNS_model_status = use_TNS_model

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            self.filename = f'{save_dir}/matrices_nfftlog{self.nfftlog}_Afull-{A_full_status}_use_TNS-{use_TNS_model_status}.npy'
        else:
            self.filename = None

        if use_TNS_model:
            print("removing Δ P(k,mu)")
            #... some people claim this violates momentum conservation and Galilean invariance!

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
            if use_TNS_model_status:
                return 0 * self.Imatrix(nu1,nu2)
            else:
                return self.Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*(nu1+nu2)))/(4*nu2*(1+nu2)*(-1+2*nu2))

        def MB2_11(nu1, nu2):
            return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2)

        def MC2_11(nu1, nu2):
            if use_TNS_model_status:
                return 0 * self.Imatrix(nu1,nu2)
            else:
                return self.Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu2*(1+nu2))

        def MD2_21(nu1, nu2):
            if use_TNS_model_status:
                return MB2_21(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((-1+2*nu1-4*nu2)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2*(-1+nu2+2*nu2**2))

        def MD3_21(nu1, nu2):
            if use_TNS_model_status:
                return MB3_21(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2))/(4*nu1*nu2*(1+nu2))

        def MD2_22(nu1, nu2):
            if use_TNS_model_status:
                return MB2_22(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*(3*(3-2*(nu1+nu2))*(1-2*(nu1+nu2)))/(32*nu1*(1+nu1)*nu2*(1+nu2))

        def MD3_22(nu1, nu2):
            if use_TNS_model_status:
                return MB3_22(nu1, nu2)
            else:
                return self.Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2)*(1+2*(nu1**2-4*nu1*nu2+nu2**2)))/(16*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))

        def MD4_22(nu1, nu2):
            if use_TNS_model_status:
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
            if use_TNS_model_status:
                return self.Imatrix(nu1, nu2)*(3*(-3 + 2*nu1)*(-3 + 2*nu1 + 2*nu2))/ (8.*nu1*nu2*(1 + nu2)*(-3 + nu1 + nu2))
            else:
                return 0.0 * self.Imatrix(nu1, nu2)

        def MB1_22(nu1, nu2):
            if use_TNS_model_status:
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
        if self.filename is not None:
            np.save(self.filename, matrices)
        return matrices

    def get_mmatrices(self):
        if self.filename and os.path.exists(self.filename):
            print(f"Loading matrices from {self.filename}")
            return np.load(self.filename, allow_pickle=True).item()

        else:
            print(
                f"Calculating and saving matrices to {self.filename}"
                if self.filename
                else "Calculating matrices (no save path specified)"
            )

        return self.calculate_matrices()

# In[7]:


class NonLinearPowerSpectrumCalculator:
    """
    A class to calculate 1-loop corrections to the linear power spectrum.

    Attributes:
        mmatrices (tuple): Set of matrices required for 1-loop computations.
        kernels (str): Choice of kernels ('EdS' or 'fk').
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
        # Construct kTout with the system NumPy to ensure these are plain numpy arrays / python scalars
        try:
            import numpy as _np_native
        except Exception:
            # Fallback: use backend numpy
            _np_native = np
        kTout_native = _np_native.geomspace(self.kminout, self.kmaxout, self.nk)
        #_np_native.exp(_np_native.linspace(_np_native.log(self.kminout), _np_native.log(self.kmaxout), self.nk))
        # Convert to the selected backend (jax.numpy or numpy)
        try:
            self.kTout = np.asarray(kTout_native)
        except Exception:
            # As a last resort, keep the native numpy array
            self.kTout = kTout_native

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

            # Use JAX-optimized version when JAX backend is active
            if backend == 'jax' and f_over_f0_EH_jax_jit is not None:
                _, _, f0 = f_over_f0_EH_jax_jit(
                    zev=self.kwargs['z'],
                    k=k,
                    OmM0=self.kwargs['Omega_m'],
                    h=self.kwargs['h'],
                    fnu=self.kwargs['fnu'],
                    Neff=self.kwargs.get('Neff', 3.044),
                    Nnu=self.kwargs.get('Nnu', 3)
                )
            else:
                # Use NumPy version
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
        if self.kernels == 'EdS':
            self.inputfkT = None
            self.f0 = self._get_f0(cosmo=cosmo, k=self.inputpkT[0])
            self.Fkoverf0 = np.ones(len(self.kTout), dtype='f8')

        else:
            if cosmo is not None and 'z' in self.kwargs:
                self.h = cosmo.h()
                self.fnu = cosmo.Omega_nu/cosmo.Omega0_m()
                self.Omega_m = cosmo.Omega0_m()
                # Use JAX-optimized version when available
                if backend == 'jax' and f_over_f0_EH_jax_jit is not None:
                    self.inputfkT = f_over_f0_EH_jax_jit(zev=self.kwargs['z'], k=self.inputpkT[0], OmM0=self.Omega_m, h=self.h, fnu=self.fnu, Neff=self.kwargs.get('Neff', 3.044), Nnu=self.kwargs.get('Nnu', 3))
                else:
                    self.inputfkT = f_over_f0_EH(zev=self.kwargs['z'], k=self.inputpkT[0], OmM0=self.Omega_m, h=self.h, fnu=self.fnu, Neff=self.kwargs.get('Neff', 3.044), Nnu=self.kwargs.get('Nnu', 3))
                self.f0 = cosmo.scale_independent_growth_factor_f(self.kwargs['z'])
            elif all(param in self.kwargs for param in ['z', 'Omega_m', 'h', 'fnu']):
                # Use JAX-optimized version when available
                if backend == 'jax' and f_over_f0_EH_jax_jit is not None:
                    self.inputfkT = f_over_f0_EH_jax_jit(zev=self.kwargs['z'], k=self.inputpkT[0], OmM0=self.kwargs['Omega_m'], h=self.kwargs['h'], fnu=self.kwargs['fnu'], Neff=self.kwargs.get('Neff', 3.044), Nnu=self.kwargs.get('Nnu', 3))
                else:
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


    def _initialize_png(self, k, pklin, cosmo=None):
        r"""Initialize the local-PNG transfer :math:`\alpha(k) = 1 / M(k)`, times :math:`k^2`.

        The table carries :math:`\alpha k^2` rather than :math:`\alpha` because that combination
        is :math:`\propto 1/T(k)`: flat where :math:`\alpha` itself diverges, which is exactly
        the low-:math:`k` region the scale-dependent bias is sensitive to.  Callers divide by
        :math:`k^2` again -- at the Alcock-Paczynski :math:`k`, which is where the table is
        interpolated, so the factor is exact.

        :math:`\alpha` is read from ``kwargs``, in order of decreasing directness: ``alpha_png``
        (an array on the *k* grid), ``pk_prim`` (likewise, e.g. ``cosmoprimo`` ``primordial.pk_k``),
        or ``A_s`` and ``n_s`` (with ``k_pivot``, in :math:`1/\mathrm{Mpc}`, and ``h``), also
        taken from *cosmo* when it provides them.  ``png_method`` selects ``'prim'`` (default) or
        ``'transfer'``; see :func:`folps.png.alpha_png`.  Absent all of these, PNG is switched
        off -- :math:`\alpha = 0`, so that ``fnl`` has no effect.

        With massive neutrinos :math:`\alpha` inherits whichever spectrum was passed as *pklin*
        -- cold dark matter + baryons, in the usual FOLPS setup -- which is the one the bias
        multiplies, so no further choice is needed here.
        """
        try:
            from .png import alpha_png, primordial_pk
        except ImportError:  # module imported as a plain script, not as a package
            from png import alpha_png, primordial_pk

        kwargs = self.kwargs
        h = kwargs.get('h', None)
        if h is None and cosmo is not None:
            h = cosmo.h()
        alpha = kwargs.get('alpha_png', None)
        if alpha is None:
            pk_prim = kwargs.get('pk_prim', None)
            if pk_prim is None:
                A_s, n_s = kwargs.get('A_s', None), kwargs.get('n_s', None)
                if A_s is None and cosmo is not None and hasattr(cosmo, 'A_s'):
                    A_s, n_s = cosmo.A_s(), cosmo.n_s()
                if A_s is not None:
                    pk_prim = primordial_pk(k, A_s=A_s, n_s=n_s, k_pivot=kwargs.get('k_pivot', 0.05),
                                            h=h, alpha_s=kwargs.get('alpha_s', 0.),
                                            beta_s=kwargs.get('beta_s', 0.))
            if pk_prim is not None:
                if h is None:
                    raise ValueError("local PNG needs 'h' in kwargs (or a cosmo), to normalise the "
                                     "primordial spectrum")
                method = kwargs.get('png_method', 'prim')
                Omega0_m = growth_factor_z = growth_factor_znorm = None
                znorm = kwargs.get('znorm', 10.)
                if method == 'transfer':
                    Omega0_m = kwargs.get('Omega_m', None)
                    growth_factor_z = kwargs.get('growth_factor_z', None)
                    growth_factor_znorm = kwargs.get('growth_factor_znorm', None)
                    if cosmo is not None:
                        if Omega0_m is None:
                            Omega0_m = cosmo.Omega0_m()
                        if growth_factor_z is None:
                            growth_factor_z = cosmo.scale_independent_growth_factor(kwargs['z'])
                        if growth_factor_znorm is None:
                            growth_factor_znorm = cosmo.scale_independent_growth_factor(znorm)
                alpha = alpha_png(k, pklin, pk_prim, h, method=method, Omega0_m=Omega0_m,
                                  growth_factor_z=growth_factor_z,
                                  growth_factor_znorm=growth_factor_znorm, znorm=znorm)
        self.has_png = alpha is not None
        if self.has_png:
            self.alphak2 = interp(self.kTout, k, np.asarray(alpha) * np.asarray(k)**2)
        else:
            self.alphak2 = np.zeros_like(self.kTout)

    def _initialize_nonwiggle_power_spectrum(self, inputpkT, pknow=None, cosmo=None,k=None):
        """
        Initializes non-wiggle linear power spectrum.
        """
        if pknow is None:
            if cosmo is not None:
                if backend == 'jax':
                    self.inputpkT_NW = get_pknow_jax(inputpkT[0], inputpkT[1], cosmo.h())
                else:
                    self.inputpkT_NW = get_pknow(inputpkT[0], inputpkT[1], cosmo.h())
            elif 'h' in self.kwargs:
                if backend == 'jax':
                    self.inputpkT_NW = get_pknow_jax(inputpkT[0], inputpkT[1], self.kwargs['h'])
                else:
                    self.inputpkT_NW = get_pknow(inputpkT[0], inputpkT[1], self.kwargs['h'])
        else:
            # self.inputpkT_NW = pknow[0], pknow[1]
            self.inputpkT_NW = extrapolate_pklin(k, pknow)



    def _initialize_liner_power_spectra(self, inputpkT):
        """
        Initializes linear power spectra for density, density-velocity and velocity fields.
        """
        if self.kernels == 'EdS':
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

        if self.kernels == 'EdS':
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
        if use_TNS_model_status:
            I2uudd_1D = self.K**3 * (np.sum(vecf @ MB1_11 * vecf, axis=-1)).real
            I2uudd_2D = self.K**3 * (np.sum(vecf @ MB2_11 * vecf, axis=-1)).real
        else:
            I2uudd_1D = self.K**3 * (np.sum(vecf @ MB1_11 * vecf, axis=-1) + np.sum(vec @ MC1_11 * vecff, axis=-1)).real
            I2uudd_2D = self.K**3 * (np.sum(vecf @ MB2_11 * vecf, axis=-1) + np.sum(vec @ MC2_11 * vecff, axis=-1)).real

        ##I2uudd_1D = self.K**3 * (np.sum(vecf @ MB1_11 * vecf, axis=-1) + np.sum(vec @ MC1_11 * vecff, axis=-1)).real
        ##I2uudd_2D = self.K**3 * (np.sum(vecf @ MB2_11 * vecf, axis=-1) + np.sum(vec @ MC2_11 * vecff, axis=-1)).real
        I3uuud_2D = self.K**3 * np.sum(vecf @ MD2_21 * vecff, axis=-1).real
        I3uuud_3D = self.K**3 * np.sum(vecf @ MD3_21 * vecff, axis=-1).real
        I4uuuu_2D = self.K**3 * np.sum(vecff @ MD2_22 * vecff, axis=-1).real
        I4uuuu_3D = self.K**3 * np.sum(vecff @ MD3_22 * vecff, axis=-1).real
        I4uuuu_4D = self.K**3 * np.sum(vecff @ MD4_22 * vecff, axis=-1).real

        #new
        I3uuud_1_B = self.K**3 * np.sum(vecf_b @ MB1_21 * vec_b, axis=1).real
        I4uuuu_1_B = self.K**3 * np.sum(vecf_b @ MB1_22 * vecf_b, axis=1).real

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
        self._initialize_png(k, pklin, cosmo=cosmo)
        self._initialize_nonwiggle_power_spectrum(inputpkT=self.inputpkT, pknow=pknow, cosmo=cosmo,k=k)
        self._initialize_liner_power_spectra(inputpkT=self.inputpkT)
        self._initialize_fftlog_terms()

        #Computations for Table
        # use backend-aware interp (from tools or tools_jax) to avoid mixing numpy and jax
        self.pk_l = interp(self.kTout, self.inputpkT[0], self.inputpkT[1])
        self.pk_l_NW = interp(self.kTout, self.inputpkT_NW[0], self.inputpkT_NW[1])


        self.sigma2w = 1 / (6 * np.pi**2) * simpson(self.inputpkTff[1], x=self.inputpkTff[0])
        self.sigma2w_NW = 1 / (6 * np.pi**2) * simpson(self.inputpkTff_NW[1], x=self.inputpkTff_NW[0])


        #rbao = 104.
        p = np.geomspace(10**(-6), 0.4, num=100)
        PSL_NW = interp(p, self.inputpkT_NW[0], self.inputpkT_NW[1])
        self.sigma2_NW = 1 / (6 * np.pi**2) * simpson(PSL_NW * (1 - spherical_jn_backend(0, p * self.rbao) + 2 * spherical_jn_backend(2, p * self.rbao)), x=p)
        self.delta_sigma2_NW = 1 / (2 * np.pi**2) * simpson(PSL_NW * spherical_jn_backend(2, p * self.rbao), x=p)
        #self.sigma2_NW = 1 / (6 * np.pi**2) * simpson(PSL_NW * (1 - special.spherical_jn(0, p * self.rbao) + 2 * special.spherical_jn(2, p * self.rbao)), x=p)
        #self.delta_sigma2_NW = 1 / (2 * np.pi**2) * simpson(PSL_NW * special.spherical_jn(2, p * self.rbao), x=p)

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

            # For JAX, use interp_at_kmin for better numerical accuracy and speed
            # For NumPy, keep the original interp approach
            if backend == 'jax':
                Pb22 = P22[5] - interp_at_kmin(self.kTout, P22[5])
                Pb2bs2 = P22[6] - interp_at_kmin(self.kTout, P22[6])
                Pb2s2 = P22[7] - interp_at_kmin(self.kTout, P22[7])
            else:
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
                I1udd_1_b2 = P22[24];    I1udd_1_bs2 = P22[27];
                I2uud_1_b2 = P22[25];    I2uud_1_bs2 = P22[28];
                I2uud_2_b2 = P22[26];    I2uud_2_bs2 = P22[29];

                common_values.extend([I1udd_1_b2, I2uud_1_b2, I2uud_2_b2,
                                      I1udd_1_bs2, I2uud_1_bs2, I2uud_2_bs2
                                    ])

            # Last of the interpolated block, so every index before it -- pk_l at 1, Fkoverf0
            # at 2 -- keeps the meaning it had before local PNG was added; see table_alphak2.
            common_values.append(self.alphak2)
            common_values.append(sigma2w)

            if extra_NW:
                common_values.extend([self.sigma2_NW, self.delta_sigma2_NW])

            common_values.append(self.f0)

            return tuple(common_values)

        self.TableOut = combine_loop_terms(self, P22, P13overpkl, self.pk_l, self.sigma2w)
        self.TableOut_NW = combine_loop_terms(self, P22_NW, P13overpkl_NW, self.pk_l_NW, self.sigma2w_NW, extra_NW=True)

        return (self.TableOut, self.TableOut_NW)

    def get_linear(self, k, pklin, pknow=None, cosmo=None, **kwargs):
        """
        Returns k_out, P_lin(k_out), P_lin_NW(k_out), and f(k_out).

        This method does NOT require calculate_loop_table().
        """

        # Store kwargs for f(k)
        self.kwargs = kwargs

        # Extrapolate input linear spectrum
        self.inputpkT = extrapolate_pklin(k, pklin)

        # Initialize f(k)/f0 and f0
        self._initialize_factors(cosmo=cosmo, k=k)

        # Initialize the local-PNG transfer alpha(k)
        self._initialize_png(k, pklin, cosmo=cosmo)

        # Initialize non-wiggle spectrum
        self._initialize_nonwiggle_power_spectrum(
            inputpkT=self.inputpkT, pknow=pknow, cosmo=cosmo, k=k
        )

        # Interpolate onto output grid
        pk_l = interp(self.kTout, self.inputpkT[0], self.inputpkT[1])
        pk_l_NW = interp(self.kTout, self.inputpkT_NW[0], self.inputpkT_NW[1])

        # Scale-dependent growth rate
        fk = self.f0 * self.Fkoverf0

        return {"k": self.kTout,"pk_l": pk_l,"pk_l_NW": pk_l_NW,"f_k": fk,"f0": self.f0,
                "alpha_png": self.alphak2 / self.kTout**2}


def split_png_pars(pars, ncore, npng):
    r"""Split a flat bias vector into ``(core, png, X_FoG)``.

    The local-PNG parameters are optional and sit between the core biases and ``X_FoG``, which
    stays last so that the ``pars[-1]`` idiom keeps working: a vector without them is read with
    every PNG parameter zero, which reproduces the Gaussian model exactly.

    Parameters
    ----------
    pars : sequence
        ``[*core, X_FoG]`` or ``[*core, *png, X_FoG]``.
    ncore, npng : int
        Number of core and PNG parameters expected.
    """
    n = len(pars)
    if n == ncore + 1:
        return pars[:ncore], (0.,) * npng, pars[-1]
    if n == ncore + 1 + npng:
        return pars[:ncore], pars[ncore:ncore + npng], pars[-1]
    raise ValueError(f"expected {ncore + 1} bias parameters, or {ncore + 1 + npng} with PNG; got {n}")


def table_alphak2(table, nowiggle=False):
    r"""Return the :math:`\alpha(k) k^2` column of a loop *table*.

    It is the last entry of the interpolated block, so that adding it left every index before it
    untouched; the entries after it are ``sigma2w``, then -- on the no-wiggle table, hence
    *nowiggle* -- ``sigma2_NW`` and ``delta_sigma2_NW``, then ``f0``.  Counting from the end
    therefore works whether or not the loop terms have been interpolated.  Both tables carry the
    same :math:`\alpha`, which is why the callers inside folps only ever read the wiggle one.
    """
    return table[-5 if nowiggle else -3]


def with_table_alphak2(table, alphak2, nowiggle=False):
    r"""Return *table* with its :math:`\alpha(k) k^2` column replaced by *alphak2*.

    For a caller that computes the local-PNG transfer itself -- from the table's own linear
    power spectrum and a primordial spectrum, say -- rather than letting
    :meth:`NonLinearPowerSpectrumCalculator._initialize_png` build it from the loop-table kwargs.
    Both tables need it: the no-wiggle one is what :meth:`get_eft_pkmu` reads for the no-wiggle
    half of the IR resummation.  See :func:`table_alphak2` for *nowiggle*.
    """
    table = list(table)
    table[-5 if nowiggle else -3] = alphak2
    return tuple(table)


def png_prim_ratio(alphai, alphaj, alphak):
    r"""Return :math:`\alpha_i \alpha_j / \alpha_k`, and zero wherever :math:`\alpha_k` is.

    This is the combination the primordial bispectrum is built from,
    :math:`M_1 M_2 M_3 P_\phi(k_i) P_\phi(k_j) = \alpha_i \alpha_j P(k_i) P(k_j) / \alpha_k`.
    A table built without PNG inputs carries :math:`\alpha = 0`, which would make the ratio
    0/0; the guard is doubled so that the derivative is finite there too.
    """
    safe = np.where(alphak > 0., alphak, 1.)
    return np.where(alphak > 0., alphai * alphaj / safe, 0.)


def weights_leggauss(nx, sym=False):
    """Gauss-Legendre nodes and weights, symmetrised onto [0, 1] when *sym*."""
    import numpy as _np_native
    x, wx = _np_native.polynomial.legendre.leggauss((1 + sym) * nx)
    if sym:
        x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
    return x, wx


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
        self._printed_model_damping_pk = False
        self._chatty = False
        #self._printed_model_damping_bk = False

    def set_bias_scheme(self, pars, bias_scheme="folps"):
        """Sets the nuisance parameters based on the selected bias scheme."""
        # Convert to lowercase to make comparison case-insensitive
        bias_scheme = bias_scheme.lower()

        if bias_scheme in ["folps", "pat", "mcdonald"]:
            if pars is None:
                pars = [1.0, 0.5, 0.3, 0.1, 0.01, 0.02, 0.03, 0.04, 0.001, 0.002, 0.003, 0.0]
            core, png, X_FoG_p = split_png_pars(pars, 11, 2)
            (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP) = core

        elif bias_scheme in ["assassi", "classpt"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for Assassi/classpt bias scheme.")
            core, png, X_FoG_p = split_png_pars(pars, 11, 2)
            (b1_classPT, b2_classPT, bG2_classPT, bGamma3_classPT, alpha0, alpha2, alpha4,
             ctilde, alphashot0, alphashot2, PshotP) = core
            b1 = b1_classPT
            b2 = b2_classPT - 4/3 * bG2_classPT
            bs2 = 2 * bG2_classPT
            b3nl = -32/21 * (bG2_classPT + 2/5 * bGamma3_classPT)

        elif bias_scheme in ["desi", "priordocument", "dr2", "priordoc"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for 'priordocument' bias scheme.")
            core, png, X_FoG_p = split_png_pars(pars, 11, 2)
            (b1_priordoc, b2_priordoc, bK2_priordoc, btd_priordoc, alpha0, alpha2, alpha4,
             ctilde, alphashot0, alphashot2, PshotP) = core
            b1 = b1_priordoc
            b2 = b2_priordoc
            bs2 = 2.* bK2_priordoc
            b3nl = -32/21 * (bK2_priordoc + 2/5 * btd_priordoc)

        else:
            raise ValueError("Invalid bias scheme. Choose from 'folps', 'pat', 'mcdonald', 'assassi', 'classpt', 'desi', 'priordocument', 'dr2', or 'priordoc' (case-insensitive).")

        # Always returned in the long, PNG-carrying convention, whatever the input length was.
        return [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP,
                png[0], png[1], X_FoG_p]


    def interp_table(self, k, table, A_full_status):
        """Interpolation of non-linear terms.

        Behavior:
        - If backend is JAX, use the JAX-friendly `interp`.
        - If backend is NumPy, use CubicSpline for interpolation.
        """

        # 29 = the interpolated block [pk_l, alpha k^2, Fkoverf0, ... I4uuuu_1B], A_full adding 6.
        extra = 6 if A_full_status else 0

        # Detect JAX backend
        is_jax_backend = getattr(np, "__name__", "numpy").startswith("jax.")

        # Columns to interpolate
        cols_to_interp = table[1:29 + extra]

        if is_jax_backend:
            # JAX backend: interpolate each column individually to avoid dimension issues
            interp_results = []
            for col in cols_to_interp:
                interp_val = interp(k, table[0], col)
                interp_results.append(interp_val)

            interp_tuple = tuple(interp_results)
            return interp_tuple + tuple(table[29 + extra:])

        else:
            # NumPy backend: use SciPy CubicSpline column by column
            from scipy.interpolate import CubicSpline

            def interp_scipy(k_query, x, y):
                return CubicSpline(x, y, extrapolate=True)(k_query)

            interp_results = []
            for col in cols_to_interp:
                interp_val = interp_scipy(k, table[0], col)
                interp_results.append(interp_val)

            interp_tuple = tuple(interp_results)
            return interp_tuple + tuple(table[29 + extra:])

    def k_ap(self, kobs, muobs, qpar, qper):
        """Return the true wave-number ‘k_AP’."""
        F = qpar / qper
        return (kobs / qper) * (1 + muobs**2 * (1. / F**2 - 1))**0.5

    def mu_ap(self, muobs, qpar, qper):
        """Return the true ‘mu_AP’."""
        F = qpar / qper
        return (muobs / F) * (1 + muobs**2 * (1 / F**2 - 1))**-0.5

    def get_eft_pkmu(self, kev, mu, pars, table, damping='lor', damping_method=None, use_GTNS=None):
        """Calculate the EFT galaxy power spectrum in redshift space.

        damping_method: which terms the FoG damping kernel multiplies (the tree-level Kaiser
        term itself is handled in get_rsd_pkmu; here it only controls the ctr/sn factors and,
        through use_GTNS=None, the GTNS default). 'tree+loop' damps the tree-level Kaiser term
        and the loop bracket only; 'tree+loop+ctr' (default; None is an alias) additionally the
        counterterms; 'tree+loop+ctr+sn' (alias 'all') additionally the shot noise.
        Legacy 'tree'/'tree-gtns' are deprecated and raise (use 'tree+loop+ctr').

        use_GTNS: whether to keep GTNS = -(k mu f0)^2 sigma2w * P_Kaiser, the perturbative FoG
        suppression of the tree-level spectrum, inside the damped loop bracket.
        None (default) follows damping_method: kept for 'loop+ctr', dropped for every
        'tree+...' method, since damping the tree level resums GTNS non-perturbatively and
        keeping both would double count at O(lambda^2). True / False override that. In
        particular damping_method='tree+loop+ctr' with use_GTNS=True is the (double-counting)
        convention used by comet's VDG_infty, which multiplies the full PT spectrum -- its own
        perturbative sigma_v^2 terms included -- by W_infty.
        The TNS model handles the FoG term itself and never adds GTNS, whatever use_GTNS says.
        """
        damping_method = _normalize_damping_method(damping_method)
        _resolve_use_gtns(use_GTNS, damping_method)  # validate early, even if the model overrides below
        core, (fnl, bphi), X_FoG_p = split_png_pars(pars, 11, 2)
        (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP) = core

        if A_full_status:
            (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
             Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
             I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D,
             I3uuud_1B, I4uuuu_1B,
             I1udd_1_b2, I2uud_1_b2, I2uud_2_b2, I1udd_1_bs2, I2uud_1_bs2, I2uud_2_bs2,
             alphak2, sigma2w, *_, f0) = table
        else:
            (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
             Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
             I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D,
             I3uuud_1B, I4uuuu_1B,
             alphak2, sigma2w, *_, f0) = table

        fk = Fkoverf0 * f0
        Pdt_L = pkl * Fkoverf0
        Ptt_L = pkl * Fkoverf0**2
        # Local-PNG scale-dependent bias: the linear response of the tracer, b1 + b_phi f_NL
        # alpha(k).  It replaces b1 in the terms that are first order in the linear bias -- the
        # tree-level Kaiser spectrum and what multiplies it -- and *not* in the 1-loop bracket,
        # whose b1 sits inside convolution integrals that alpha(k) cannot be pulled out of.
        b1eff = b1 + fnl * bphi * alphak2 / kev**2

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
            # _use_gtns is bound below, after the model dispatch may have overridden damping_method;
            # this closure only runs from PloopSPTs, which is called after that point.
            if use_TNS_model_status or not _use_gtns:
                return 0
            else:
                return -((kev * mu * f0)**2 * sigma2w * (b1eff**2 * pkl + 2 * b1eff * f0 * mu**2 * Pdt_L + f0**2 * mu**4 * Ptt_L))

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
            return (b1eff + mu**2 * fk)**2 * pkl

        def PctNLOs(mu, b1, ctilde):
            return ctilde * (mu * kev * f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)

        def Pcts(mu, alpha0, alpha2, alpha4):
            return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4) * kev**2 * pkl

        def Pshot(mu, alphashot0, alphashot2, PshotP):
            return PshotP * (alphashot0 + alphashot2 * (kev * mu)**2)

        # --- Model self.model ---
        if not getattr(self, '_printed_model_damping_pk', False):
            if self.model == "EFT" and damping is not None:
                print(f"[FOLPS] Model Pk: {self.model}, damping: None (WARNING: EFT does not use damping, ignoring damping)")
            else:
                if self._chatty:
                    print(f"[FOLPS] Model Pk: {self.model}, Damping: {damping}")

            self._printed_model_damping_pk = True

        if self.model == "EFT":
            # EFT ignores FoG damping entirely: tree-level Kaiser undamped, GTNS kept.
            damping = None
            damping_method = 'loop+ctr'
        elif self.model == "TNS":
            if not use_TNS_model_status:
                raise RuntimeError("[FOLPS] To use the TNS model, you must set use_TNS_model=True in MatrixCalculator.")
        elif self.model == "FOLPSD":
            if damping is None:
                print("[FOLPS] For FOLPSD you must specify a damping ('exp', 'lor', 'vdg'). Default: 'lor'.")
                damping = 'lor'
        # Resolved here, not at the top: the model dispatch above may have overridden damping_method.
        _use_gtns = _resolve_use_gtns(use_GTNS, damping_method)
        W = fog_damping((kev * mu, X_FoG_p), (kev * mu, X_FoG_p), f=f0, sigma2v=sigma2w, damping=damping)
        # Which terms the kernel multiplies: loop bracket always; ctr/sn if named in damping_method.
        W_ctr = 1. if 'ctr' not in damping_method else W
        W_sn = 1. if 'sn' not in damping_method else W

        PK = W * PloopSPTs(mu, b1, b2, bs2, b3nl) + W_sn * Pshot(mu, alphashot0, alphashot2, PshotP)

        return PK + W_ctr * (Pcts(mu, alpha0, alpha2, alpha4) + PctNLOs(mu, b1, ctilde))

    def get_rsd_pkmu(self, k, mu, pars, table, table_now, IR_resummation=True, damping='lor',
                     damping_method=None, use_GTNS=None):
        """Return redshift space P(k, mu) given input tables.

        damping_method: which terms the FoG damping kernel multiplies. 'tree+loop' damps
        the tree-level Kaiser term and the loop bracket only; 'tree+loop+ctr' (default;
        None is an alias) additionally the counterterms; 'tree+loop+ctr+sn' (alias 'all')
        additionally the shot noise. Legacy 'tree'/'tree-gtns' are deprecated and raise (use
        'tree+loop+ctr'). The EFT model ignores damping_method (no damping, GTNS kept).

        use_GTNS: whether to keep the perturbative GTNS term; None (default) follows
        damping_method (kept for 'loop+ctr', dropped for 'tree+...'), True / False override it.
        See :meth:`get_eft_pkmu`. Note use_GTNS is orthogonal to the tree-level damping, so
        damping_method='tree+loop+ctr' with use_GTNS=True damps the tree *and* keeps GTNS.
        """
        damping_method = _normalize_damping_method(damping_method)
        if self.model == "EFT":
            # EFT ignores FoG damping entirely (see get_eft_pkmu): tree-level Kaiser undamped, GTNS kept.
            damping_method = 'loop+ctr'
        table = self.interp_table(k, table, A_full_status)
        table_now = self.interp_table(k, table_now, A_full_status)
        (fnl, bphi) = split_png_pars(pars, 11, 2)[1]
        b1 = pars[0] + fnl * bphi * table_alphak2(table) / k**2
        f0 = table[-1]
        fk = table[1] * f0
        pkl, pkl_now = table[0], table_now[0]
        sigma2, delta_sigma2 = table_now[-3:-1]
        W_tree = 1.
        if damping_method.startswith('tree'):
            # Damping of the tree-level Kaiser term: same kernel as get_eft_pkmu's, with the
            # wiggle table's sigma2w (table trailing entries are [..., sigma2w, f0]).
            if damping not in ('exp', 'lor', 'vdg'):
                raise ValueError(f"damping_method={damping_method!r} requires damping 'exp', 'lor' or 'vdg', got {damping!r}")
            X_FoG_p = pars[-1]
            sigma2w = table[-2]
            W_tree = fog_damping((k * mu, X_FoG_p), (k * mu, X_FoG_p), f=f0, sigma2v=sigma2w, damping=damping)
        # Sigma² tot for IR-resummations, see eq.~ 3.59 at arXiv:2208.02791
        if IR_resummation:
            sigma2t = (1 + f0*mu**2 * (2 + f0))*sigma2 + (f0*mu)**2 * (mu**2 - 1) * delta_sigma2
        else:
            sigma2t = 0
        pkmu = (W_tree * (b1 + fk * mu**2)**2 * (pkl_now + np.exp(-k**2 * sigma2t)*(pkl - pkl_now)*(1 + k**2 * sigma2t))
                 + np.exp(-k**2 * sigma2t) * self.get_eft_pkmu(k, mu, pars, table, damping, damping_method=damping_method, use_GTNS=use_GTNS)
                 + (1 - np.exp(-k**2 * sigma2t)) * self.get_eft_pkmu(k, mu, pars, table_now, damping, damping_method=damping_method, use_GTNS=use_GTNS))
        return pkmu

    def get_rsd_pkell(self, kobs, qpar, qper, pars, table, table_now,
                      bias_scheme="folps", damping='lor', nmu=6, ells=(0, 2, 4), IR_resummation=True,
                      damping_method=None, use_GTNS=None):
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
            damping_method (str): Which terms the FoG kernel multiplies; see :meth:`get_rsd_pkmu`.
            use_GTNS (bool): Whether to keep the perturbative GTNS term; see :meth:`get_eft_pkmu`.

        Returns:
            array: Power spectrum multipoles for each ell.
        """
        pars = self.set_bias_scheme(pars, bias_scheme=bias_scheme)
        muobs, wmu = weights_leggauss(nmu, sym=True)
        wmu = np.array([wmu * (2 * ell + 1) * legendre(ell)(muobs) for ell in ells])
        jac, kap, muap = (qpar * qper**2)**(-1), self.k_ap(kobs[:, None], muobs, qpar, qper), self.mu_ap(muobs, qpar, qper)[None, :]
        pkmu = jac * self.get_rsd_pkmu(kap, muap, pars, table, table_now, IR_resummation, damping,
                                       damping_method=damping_method, use_GTNS=use_GTNS)
        return np.sum(pkmu * wmu[:, None, :], axis=-1)

    def get_eft_pkmu_monomials(self, kev, mu, table, damping_method=None, use_GTNS=None):
        r""":meth:`get_eft_pkmu` decomposed over bias monomials.

        Returns ``(damped, undamped, sigma2w)``, two :class:`BiasPolynomial` and the velocity
        dispersion the kernel needs, such that for any bias vector

        ``get_eft_pkmu(pars) == W * damped.evaluate(pars) + undamped.evaluate(pars)``

        with ``W`` the Finger-of-God kernel built from *table*'s own ``sigma2w``.  Which blocks
        sit in which part follows ``damping_method`` exactly as in :meth:`get_eft_pkmu`: the loop
        bracket is always damped, the counterterms and the shot noise only when named.

        Line by line the same expressions, with the biases symbolic.  The terms that look
        non-polynomial are not: ``ATNS = b1**3 Af(mu, f0/b1)`` and ``DRSD = b1**4 Df(mu, f0/b1)``
        carry a compensating power of ``f0/b1`` in every term.
        """
        damping_method = _normalize_damping_method(damping_method)
        use_gtns = _resolve_use_gtns(use_GTNS, damping_method)

        if A_full_status:
            (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
             Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
             I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D,
             I3uuud_1B, I4uuuu_1B,
             I1udd_1_b2, I2uud_1_b2, I2uud_2_b2, I1udd_1_bs2, I2uud_1_bs2, I2uud_2_bs2,
             alphak2, sigma2w, *_, f0) = table
        else:
            (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
             Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
             I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D,
             I3uuud_1B, I4uuuu_1B,
             alphak2, sigma2w, *_, f0) = table

        fk = Fkoverf0 * f0
        Pdt_L = pkl * Fkoverf0
        Ptt_L = pkl * Fkoverf0**2
        b1 = BiasPolynomial.variable('b1')
        b2 = BiasPolynomial.variable('b2')
        bs2 = BiasPolynomial.variable('bs')
        b3nl = BiasPolynomial.variable('b3nl')
        # b1 + b_phi f_NL alpha(k), see get_eft_pkmu; Gaussian runs never evaluate the extra
        # monomials, they only carry them.
        b1eff = b1 + BiasPolynomial.variable('fnl') * BiasPolynomial.variable('bphi', alphak2 / kev**2)

        PddXloop = (b1 * b1 * Ploop_dd + b1 * b2 * (2 * Pb1b2) + b1 * bs2 * (2 * Pb1bs2)
                    + b2 * b2 * Pb22 + b2 * bs2 * (2 * Pb2bs2) + bs2 * bs2 * Pb2s2
                    + b1 * b3nl * (2 * sigma23pkl))
        PdtXloop = b1 * Ploop_dt + b2 * Pb2t + bs2 * Pbs2t + b3nl * (Fkoverf0 * sigma23pkl)
        PttXloop = BiasPolynomial.constant(Ploop_tt)

        # b1**3 Af(mu, f0/b1): the f0**n term carries b1**(3-n).
        ATNS = (b1 * b1 * (f0 * mu**2 * I1udd_1)
                + b1 * (f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2))
                + BiasPolynomial.constant(f0**3 * (mu**4 * I3uuu_2 + mu**6 * I3uuu_3)))
        # b1**4 Df(mu, f0/b1), likewise.
        DRSD = (b1 * b1 * (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D))
                + b1 * (f0**3 * (mu**2 * I3uuud_1B + mu**4 * I3uuud_2D + mu**6 * I3uuud_3D))
                + BiasPolynomial.constant(f0**4 * (mu**2 * I4uuuu_1B + mu**4 * I4uuuu_2D
                                                   + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D)))
        PloopSPTs = PddXloop + PdtXloop * (2 * f0 * mu**2) + PttXloop * (mu**4 * f0**2) + ATNS + DRSD
        if A_full_status:
            # b1**3 Af_b2(mu, f0/b1) b2 / (2 b1) and the bs2 counterpart.
            for coefficient, terms in [(b2, (I1udd_1_b2, I2uud_1_b2, I2uud_2_b2)),
                                       (bs2, (I1udd_1_bs2, I2uud_1_bs2, I2uud_2_bs2))]:
                first, second, third = terms
                PloopSPTs = PloopSPTs + coefficient * (b1 * (f0 * mu**2 * first / 2)
                                                      + BiasPolynomial.constant(
                                                          f0**2 * (mu**2 * second + mu**4 * third) / 2))
        if not (use_TNS_model_status or not use_gtns):
            gtns = -((kev * mu * f0)**2 * sigma2w)
            PloopSPTs = PloopSPTs + (b1eff * b1eff * (gtns * pkl)
                                     + b1eff * (gtns * 2 * f0 * mu**2 * Pdt_L)
                                     + BiasPolynomial.constant(gtns * f0**2 * mu**4 * Ptt_L))

        # (b1eff + mu**2 fk)**2 pkl
        Z1 = b1eff + BiasPolynomial.constant(mu**2 * fk)
        PKaiserLs = Z1 * Z1 * pkl
        PctNLOs = PKaiserLs * BiasPolynomial.variable('ctilde', (mu * kev * f0)**4 * sigma2w**2)
        Pcts = (BiasPolynomial.variable('alpha0', kev**2 * pkl)
                + BiasPolynomial.variable('alpha2', mu**2 * kev**2 * pkl)
                + BiasPolynomial.variable('alpha4', mu**4 * kev**2 * pkl))
        # Normalisation-free: PshotP multiplies sn0 / sn2 in get_rsd_pkell_from_monomials instead.
        Pshot = (BiasPolynomial.variable('sn0', np.ones_like(kev * mu))
                 + BiasPolynomial.variable('sn2', (kev * mu)**2))

        damped, undamped = PloopSPTs, BiasPolynomial()
        for block, flag in [(Pcts + PctNLOs, 'ctr'), (Pshot, 'sn')]:
            if flag in damping_method:
                damped = damped + block
            else:
                undamped = undamped + block
        # Returned rather than read off a trailing index by the caller: the no-wiggle table has
        # [sigma2w_NW, sigma2_NW, delta_sigma2_NW, f0] at the end, so table[-2] is *not* sigma2w
        # there.  The positional unpacking above gets it right for both.
        return damped, undamped, sigma2w

    def get_rsd_pkmu_monomials(self, k, mu, table, table_now, IR_resummation=True,
                               damping_method=None, use_GTNS=None):
        r""":meth:`get_rsd_pkmu` decomposed over bias monomials.

        Returns ``(damped_wiggle, damped_nowiggle, undamped, lam, sigma2w, sigma2w_nowiggle)``: three
        :class:`BiasPolynomial` and the collocation variable
        :math:`\Lambda = f_0^2 (k\mu)^2`, such that

        ``get_rsd_pkmu(pars) == W[sigma2w] * damped_wiggle + W[sigma2w_NW] * damped_nowiggle + undamped``

        evaluated at *pars*.  Two damped blocks rather than one because the wiggle and no-wiggle
        halves of the IR resummation carry different :math:`\sigma_w^2`; they share one
        :math:`\Lambda` grid, since that scalar enters the kernel at evaluation time.
        """
        damping_method = _normalize_damping_method(damping_method)
        if self.model == 'EFT':
            damping_method = 'loop+ctr'
        table = self.interp_table(k, table, A_full_status)
        table_now = self.interp_table(k, table_now, A_full_status)
        f0 = table[-1]
        fk = table[1] * f0
        pkl, pkl_now = table[0], table_now[0]
        sigma2, delta_sigma2 = table_now[-3:-1]
        sigma2t = ((1 + f0 * mu**2 * (2 + f0)) * sigma2 + (f0 * mu)**2 * (mu**2 - 1) * delta_sigma2
                   if IR_resummation else 0)
        wiggle_weight = np.exp(-k**2 * sigma2t)

        b1eff = (BiasPolynomial.variable('b1')
                 + BiasPolynomial.variable('fnl')
                 * BiasPolynomial.variable('bphi', table_alphak2(table) / k**2))
        Z1 = b1eff + BiasPolynomial.constant(fk * mu**2)
        tree = Z1 * Z1 * (pkl_now + wiggle_weight * (pkl - pkl_now) * (1 + k**2 * sigma2t))

        damped_wiggle, undamped_wiggle, sigma2w = self.get_eft_pkmu_monomials(
            k, mu, table, damping_method=damping_method, use_GTNS=use_GTNS)
        damped_nowiggle, undamped_nowiggle, sigma2w_nowiggle = self.get_eft_pkmu_monomials(
            k, mu, table_now, damping_method=damping_method, use_GTNS=use_GTNS)

        damped_wiggle = damped_wiggle * wiggle_weight
        damped_nowiggle = damped_nowiggle * (1. - wiggle_weight)
        undamped = undamped_wiggle * wiggle_weight + undamped_nowiggle * (1. - wiggle_weight)
        # The tree-level Kaiser term is damped with the *wiggle* table's sigma2w, so it joins the
        # wiggle block; otherwise it is undamped.
        if damping_method.startswith('tree'):
            damped_wiggle = damped_wiggle + tree
        else:
            undamped = undamped + tree
        lam = f0**2 * (k * mu)**2
        return damped_wiggle, damped_nowiggle, undamped, lam, sigma2w, sigma2w_nowiggle

    def get_rsd_pkmu_monomials_tables(self, kobs, qpar, qper, table, table_now, nmu=6, ells=(0, 2, 4),
                                      IR_resummation=True, damping_method=None, use_GTNS=None,
                                      fixed_bias=None):
        r"""Bias-monomial tables on the ``(k, mu)`` quadrature grid, with the damping left unapplied.

        Multiplying by the Finger-of-God kernel is a linear operator on the mu quadrature, which
        has only ``nmu`` nodes, so keeping that axis represents the ``X_FoG`` dependence
        **exactly** -- no collocation, no node count to tune.  (Equivalently, a ``(ell, ell', k)``
        matrix in the Legendre basis; the node basis is the same statement with the operator
        diagonal.)

        This works because the k axis survives.  The bispectrum has to collocate in ``Lambda``
        (:meth:`BispectrumCalculator.Sugiyama_Bell_monomials`) for the opposite reason: its
        orientation quadrature has 800 nodes per pair against 6 here, so keeping that axis would
        be ~400x larger rather than smaller -- and folding the window into an emulator's
        coefficients, which is what makes the bispectrum tables affordable, removes the very axis
        the damping depends on.  Here the theory grid is compacted instead
        (`full_shape.tools.rebin_spectrum2_window`) and the window is applied per call.

        Returns
        -------
        dict
            ``'monomials'``; ``'undamped'``, ``'damped_wiggle'``, ``'damped_nowiggle'`` of shape
            ``(n_monomials, n_k, nmu)``; ``'lam'`` ``(n_k, nmu)``; ``'sigma2w'``,
            ``'sigma2w_nowiggle'``; ``'legendre_weights'`` ``(n_ells, nmu)`` and ``'ells'``.
        """
        muobs, weights_mu = weights_leggauss(nmu, sym=True)
        legendre_weights = np.array([weights_mu * (2 * ell + 1) * legendre(ell)(muobs) for ell in ells])
        jac = (qpar * qper**2)**(-1)
        kap = self.k_ap(np.asarray(kobs)[:, None], muobs, qpar, qper)
        muap = self.mu_ap(muobs, qpar, qper)[None, :]

        damped_wiggle, damped_nowiggle, undamped, lam, sigma2w, sigma2w_nowiggle = \
            self.get_rsd_pkmu_monomials(kap, muap, table, table_now, IR_resummation=IR_resummation,
                                        damping_method=damping_method, use_GTNS=use_GTNS)
        if fixed_bias:
            damped_wiggle = damped_wiggle.substitute(fixed_bias)
            damped_nowiggle = damped_nowiggle.substitute(fixed_bias)
            undamped = undamped.substitute(fixed_bias)
        monomials = sorted(set(damped_wiggle.monomials()) | set(damped_nowiggle.monomials())
                           | set(undamped.monomials()))
        # The AP jacobian is constant, so it rides with the tables rather than the contraction.
        shape = np.shape(kap * muap)
        stacked = {name: polynomial.stack(monomials, shape=shape) * jac for name, polynomial in
                   [('damped_wiggle', damped_wiggle), ('damped_nowiggle', damped_nowiggle),
                    ('undamped', undamped)]}
        return {'monomials': monomials, 'lam': lam, 'sigma2w': sigma2w,
                'sigma2w_nowiggle': sigma2w_nowiggle, 'legendre_weights': legendre_weights,
                'ells': tuple(ells), **stacked}

    def get_rsd_pkell_from_monomials(self, tables, pars, damping='lor', window_operator=None):
        """Contract ``(k, mu)`` monomial *tables* from
        :meth:`RSDMultipolesPowerSpectrumCalculator.get_rsd_pkmu_monomials_tables` with the biases.

        Exact in ``X_FoG``: the kernel is evaluated at the quadrature nodes rather than interpolated.

        Without *window_operator* the result is the multipoles, ``(n_ells, n_k)``.  With it -- the
        window matrix already composed with the Legendre weights, shape ``(n_data, n_k, nmu)`` -- the
        result is the flat data vector.  The operator cannot be folded into the tables beforehand,
        since the damping sits between it and them; but it is cosmology-independent, so it is built
        once and applied here.
        """
        # set_bias_scheme returns the folps convention, [b1, b2, bs2, b3nl, alpha0, alpha2,
        # alpha4, ctilde, alphashot0, alphashot2, PshotP, (fnl, bphi,) X_FoG]; zip stops at the
        # monomial variables, leaving PshotP and X_FoG to be handled separately.
        names = ('b1', 'b2', 'bs', 'b3nl', 'alpha0', 'alpha2', 'alpha4', 'ctilde', 'sn0', 'sn2')
        core, (fnl, bphi), X_FoG = split_png_pars(pars, 11, 2)
        bias_values = dict(zip(names, core), fnl=fnl, bphi=bphi)
        # Pshot = PshotP (alphashot0 + alphashot2 (k mu)^2): folding the normalisation into the
        # two stochastic values keeps the tables free of it, so they do not have to be rebuilt
        # when the number density or the prior basis changes.
        bias_values['sn0'] = bias_values['sn0'] * core[10]
        bias_values['sn2'] = bias_values['sn2'] * core[10]
        monomial_values = bias_monomial_values(bias_values, tables['monomials'])
        total = tables['undamped']
        for suffix, sigma2v in [('wiggle', tables['sigma2w']), ('nowiggle', tables['sigma2w_nowiggle'])]:
            kernel = 1. - fog_damping_correction(X_FoG, tables['lam'], sigma2v=sigma2v,
                                                 damping=damping, nlegs=2)
            total = total + kernel * tables['damped_' + suffix]
        pkmu = np.tensordot(monomial_values, total, axes=([0], [0]))   # (n_k, nmu)
        if window_operator is None:
            return np.einsum('lm,km->lk', tables['legendre_weights'], pkmu)
        return np.einsum('dkm,km->d', window_operator, pkmu)

def get_rsd_pkell_marg_const(
    kobs, qpar, qper, pars, table, table_now,
    bias_scheme="folps", damping='lor', nmu=6, ells=(0, 2, 4), IR_resummation=True,
    model='EFT'
    ):
    """
    Computes the multipoles of the RSD power spectrum marginalizing over EFT and stochastic parameters.
    """

    multipoles = RSDMultipolesPowerSpectrumCalculator(model=model)

    def get_rsd_pkmu_const(k, mu, pars, table, table_now, IR_resummation, damping):
        # Apply bias scheme
        pars = multipoles.set_bias_scheme(pars, bias_scheme=bias_scheme)
        core, (fnl, bphi), X_FoG_p = split_png_pars(pars, 11, 2)
        (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4,
         ctilde, alphashot0, alphashot2, PshotP) = core

        # Interpolate tables
        table_interp = multipoles.interp_table(k, table, A_full_status)
        table_now_interp = multipoles.interp_table(k, table_now, A_full_status)

        # Set EFT and stochastic parameters to zero (marginalization)
        alpha0 = alpha2 = alpha4 = alphashot0 = alphashot2 = 0.0

        pars_const = (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4,
                      ctilde, alphashot0, alphashot2, PshotP, fnl, bphi, X_FoG_p)

        f0 = table_interp[-1]
        fk = table_interp[1] * f0
        # Local-PNG scale-dependent bias, as in get_rsd_pkmu.
        b1 = b1 + fnl * bphi * table_alphak2(table_interp) / k**2
        pkl, pkl_now = table_interp[0], table_now_interp[0]
        sigma2, delta_sigma2 = table_now_interp[-3], table_now_interp[-2]

        # Total Sigma^2 for IR-resummation, see eq. 3.59 in arXiv:2208.02791
        if IR_resummation:
            sigma2t = (1 + f0 * mu**2 * (2 + f0)) * sigma2 + (f0 * mu)**2 * (mu**2 - 1) * delta_sigma2
        else:
            sigma2t = 0.0

        # Compute pkmu. The tree-level Kaiser term below is undamped and the analytic
        # derivatives assume the original FOLPSD convention: pin damping_method='loop+ctr'
        # (hence use_GTNS=None -> GTNS kept). This path does not honour a use_GTNS override.
        pkmu = (
            (b1 + fk * mu**2)**2 * (pkl_now + np.exp(-k**2 * sigma2t) * (pkl - pkl_now) * (1 + k**2 * sigma2t))
            + np.exp(-k**2 * sigma2t) * multipoles.get_eft_pkmu(k, mu, pars_const, table_interp, damping, damping_method='loop+ctr')
            + (1 - np.exp(-k**2 * sigma2t)) * multipoles.get_eft_pkmu(k, mu, pars_const, table_now_interp, damping, damping_method='loop+ctr')
        )
        return pkmu

    def get_rsd_pkell(
        kobs, qpar, qper, pars, table, table_now,
        bias_scheme="folps", damping='lor', nmu=6, ells=(0, 2, 4), IR_resummation=True,
        model='EFT'
    ):
        """
        Computes the multipoles of the power spectrum.
        """
        pars = multipoles.set_bias_scheme(pars, bias_scheme=bias_scheme)

        def weights_leggauss(nx, sym=False):
            """Return weights for Gauss-Legendre integration.

            Use SciPy's roots_legendre so this function works when the
            global ``np`` is ``jax.numpy`` (which doesn't expose
            ``polynomial.legendre.leggauss``).
            The outputs are converted to the currently selected backend
            array via ``np.asarray`` so downstream code can operate with
            either NumPy or JAX arrays.
            """
            # scipy.special.roots_legendre returns (roots, weights)
            x, wx = special.roots_legendre((1 + sym) * nx)
            if sym:
                x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
            # Convert to backend array (jax.numpy or numpy)
            try:
                return np.asarray(x), np.asarray(wx)
            except Exception:
                # If conversion to backend arrays fails (e.g. unexpected backend),
                # just return the SciPy/NumPy arrays as-is.
                return x, wx

        muobs, wmu = weights_leggauss(nmu, sym=True)
        wmu_arr = np.array([wmu * (2 * ell + 1) * legendre(ell)(muobs) for ell in ells])
        jac = (qpar * qper**2)**(-1)
        kap = multipoles.k_ap(kobs[:, None], muobs, qpar, qper)
        muap = multipoles.mu_ap(muobs, qpar, qper)[None, :]

        pkmu = jac * get_rsd_pkmu_const(kap, muap, pars, table, table_now, IR_resummation, damping)
        pkell = np.sum(pkmu * wmu_arr[:, None, :], axis=-1)
        return pkell

    return get_rsd_pkell(
        kobs, qpar, qper, pars, table, table_now,
        bias_scheme=bias_scheme, damping=damping,
        nmu=nmu, ells=ells, IR_resummation=IR_resummation,
        model=model
    )



def PEFTs_derivatives(k, mu, pkl, PshotP):
    """
    Derivatives of PEFTs with respect to the EFT and stochastic parameters.

    Args:
        k: wave-number coordinates of evaluation.
        mu: cosine angle between the wave-vector ‘vec-k’ and the line-of-sight direction ‘hat-n’.
        pkl: linear power spectrum.
        PshotP: stochastic nuisance parameter.
    Returns:
        ∂P_EFTs/∂α_i with: α_i = {alpha0, alpha2, alpha4, alphashot0, alphashot2}
    """

    k2 = k**2
    k2mu2 = k2 * mu**2
    k2mu4 = k2mu2 * mu**2

    PEFTs_alpha0 = k2 * pkl
    PEFTs_alpha2 = k2mu2 * pkl
    PEFTs_alpha4 = k2mu4 * pkl
    PEFTs_alphashot0 = PshotP
    PEFTs_alphashot2 = k2mu2 * PshotP

    return (PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2)



def get_rsd_pkell_marg_derivatives(
    kobs, qpar, qper, pars, table, table_now,
    bias_scheme="folps", damping='lor', nmu=6, ells=(0, 2, 4), IR_resummation=True,
    model='EFT'
    ):
    """
    Redshift space power spectrum multipoles 'derivatives': Pℓ,i=∂Pℓ/∂α_i
    (derivatives with respect to the EFT and stochastic parameters).
    """
    multipoles = RSDMultipolesPowerSpectrumCalculator(model=model)

    def get_rsd_pkmu_derivatives(k, mu, pars, table, table_now, IR_resummation, damping):
        pars = multipoles.set_bias_scheme(pars, bias_scheme=bias_scheme)
        core, (fnl, bphi), X_FoG_p = split_png_pars(pars, 11, 2)
        (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4,
         ctilde, alphashot0, alphashot2, PshotP) = core

        kap = multipoles.k_ap(k, mu, qpar, qper)
        muap = multipoles.mu_ap(mu, qpar, qper)

        table_interp = multipoles.interp_table(kap, table, A_full_status)
        table_now_interp = multipoles.interp_table(kap, table_now, A_full_status)

        f0 = table_interp[-1]
        fk = table_interp[1] * f0
        pkl, pkl_now = table_interp[0], table_now_interp[0]
        sigma2, delta_sigma2 = table_now_interp[-3:-1]

        PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2 = PEFTs_derivatives(kap, muap, pkl, PshotP)
        PEFTs_alpha0_NW, PEFTs_alpha2_NW, PEFTs_alpha4_NW, PEFTs_alphashot0_NW, PEFTs_alphashot2_NW = PEFTs_derivatives(kap, muap, pkl_now, PshotP)

        if IR_resummation:
            sigma2t = (1 + f0 * mu**2 * (2 + f0)) * sigma2 + (f0 * mu)**2 * (mu**2 - 1) * delta_sigma2
        else:
            sigma2t = 0.0

        exp_term = np.exp(-kap**2 * sigma2t)
        exp_term_inv = 1 - exp_term

        #computing PIRs_derivatives for EFT and stochastic parameters
        PIRs_alpha0 = exp_term * PEFTs_alpha0 + exp_term_inv * PEFTs_alpha0_NW
        PIRs_alpha2 = exp_term * PEFTs_alpha2 + exp_term_inv * PEFTs_alpha2_NW
        PIRs_alpha4 = exp_term * PEFTs_alpha4 + exp_term_inv * PEFTs_alpha4_NW
        PIRs_alphashot0 = exp_term * PEFTs_alphashot0 + exp_term_inv * PEFTs_alphashot0_NW
        PIRs_alphashot2 = exp_term * PEFTs_alphashot2 + exp_term_inv * PEFTs_alphashot2_NW

        #print("======= mu =======")
        #print(muap)
        #print("======= end: mu =======")

        #print("========= results ==========")
        #print(PIRs_alpha0, PIRs_alpha2, PIRs_alpha4, PIRs_alphashot0, PIRs_alphashot2)
        #print("======= end: mu =======")

        return (PIRs_alpha0, PIRs_alpha2, PIRs_alpha4, PIRs_alphashot0, PIRs_alphashot2)

    Nx = nmu
    xGL, wGL = scipy.special.roots_legendre(Nx)

    def ModelPkl_derivatives(table, table_now, ell):
        factor = (2*ell+1)/2
        result = 1/(qper**2 * qpar) * sum(
            factor * wGL[ii] * np.array(get_rsd_pkmu_derivatives(kobs, xGL[ii], pars, table, table_now, IR_resummation, damping)) * scipy.special.eval_legendre(ell, xGL[ii])
            for ii in range(Nx)
        )
        return result

    return tuple(ModelPkl_derivatives(table, table_now, ell) for ell in ells)



#Marginalization matrices
def startProduct(A, B, invCov):
    '''Computes: A @ InvCov @ B^{T}, where 'T' means transpose.

    Args:
         A: first vector, array of the form 1 x n
         B: second vector, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n

    Returns:
         The result of: A @ InvCov @ B^{T}
    '''

    return A @ invCov @ B.T


def compute_L0(Pl_const, Pl_data, invCov, mu_prior = 0, sigma_prior = np.inf):
    '''Computes the term L0 of the marginalized Likelihood.

    Args:
         Pl_const: model multipoles for the constant part (Pℓ,const = Pℓ(α->0)), array of the form 1 x n
         Pl_data: data multipoles, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n

    Return:
         Loglikelihood for the constant part of the model multipoles
    '''

    D_const = Pl_const - Pl_data

    L0 = -0.5 * startProduct(D_const, D_const, invCov)

    # Adding prior to L0
    if isinstance(sigma_prior, (int, float)):
        mu_prior2 = np.dot(np.array(mu_prior), np.array(mu_prior))
        L0 += -0.5 * (mu_prior2 / (sigma_prior ** 2) )
    else:
        mu_prior2 = np.dot(np.array(mu_prior), np.array(mu_prior))
        sigma_prior2 = np.dot(np.array(sigma_prior), np.array(sigma_prior))
        L0 += -0.5 * (mu_prior2 / sigma_prior2)

    return L0


def compute_L1i(Pl_i, Pl_const, Pl_data, invCov, mu_prior = 0, sigma_prior = np.inf):
    '''Computes the term L1i of the marginalized Likelihood.

    Args:
         Pl_i: array with the derivatives of the power spectrum multipoles with respect to
               the EFT and stochastic parameters, i.e., Pℓ,i=∂Pℓ/∂α_i , i = 1,..., ndim
               array of the form ndim x n
         Pl_const: model multipoles for the constant part (Pℓ,const = Pℓ(α->0)), array of the form 1 x n
         Pl_data: data multipoles, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n
    Return:
         array for L1i
    '''

    D_const = Pl_const - Pl_data

    #ndim = len(Pl_i)

    #computing L1i
    #L1i = np.zeros(ndim)

    #for ii in range(ndim):
    #    term1 = startProduct(Pl_i[ii], D_const, invCov)
    #    term2 = startProduct(D_const, Pl_i[ii], invCov)
    #    L1i[ii] = -0.5 * (term1 + term2)

    L1i = - startProduct(Pl_i, D_const, invCov)

    # Adding prior to L1i
    if isinstance(sigma_prior, (int, float)):
        L1i += np.array(mu_prior) / (sigma_prior ** 2)
    else:
        L1i += np.array(mu_prior) / np.array(sigma_prior) ** 2

    return L1i


def compute_L2ij(Pl_i, invCov, sigma_prior = np.inf):
    '''Computes the term L2ij of the marginalized Likelihood.

    Args:
         Pl_i: array with the derivatives of the power spectrum multipoles with respect to
               the EFT and stochastic parameters, i.e., Pℓ,i=∂Pℓ/∂α_i , i = 1,..., ndim
               array of the form ndim x n
         invCov: inverse of covariance matrix, array of the form n x n
    Return:
         array for L2ij
    '''

    #ndim = len(Pl_i)

    #Computing L2ij
    #L2ij = np.zeros((ndim, ndim))

    #for ii in range (ndim):
        #for jj in range (ndim):
            #L2ij[ii, jj] = startProduct(Pl_i[ii], Pl_i[jj], invCov)

    L2ij = startProduct(Pl_i, Pl_i, invCov)

    # Adding prior variances to L2ij
    if isinstance(sigma_prior, (int, float)):
        L2ij += 1 / (sigma_prior ** 2)
    else:
        L2ij += np.diag(1 / np.array(sigma_prior) ** 2)

    return L2ij






#### Bispectrum




def get_f0(z, OmM0):
    """
    Compute the linear growth rate f0 = d ln D+ / d ln a at redshift z.
    This version is backend-agnostic and works with both NumPy and JAX.

    Args:
        z: redshift
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)

    Returns:
        f0: linear growth rate
    """

    def OmM(eta):
        return 1./(1. + ((1-OmM0)/OmM0)* np.exp(3*eta))

    def f1(eta):
        return 2. - 3./2. * OmM(eta)

    def f2(eta):
        return 3./2. * OmM(eta)

    etaini = -6  # initial eta, early enough to evolve as EdS (D+ ∝ a)
    zfin = z

    def etaofz(z):
        return np.log(1./(1. + z))

    etafin = etaofz(zfin)

    # Choose odeint depending on backend
    use_jax = (backend == 'jax')

    if use_jax:
        from jax.experimental.ode import odeint as jax_odeint
    else:
        from scipy.integrate import odeint as scipy_odeint

    # Differential eqs.
    def Deqs(y, eta_val):
        # y is [D, Dprime]
        D = y[0]
        Dprime = y[1]
        return np.array([Dprime, f2(eta_val)*D - f1(eta_val)*Dprime])

    # eta range and initial conditions
    eta_arr = np.linspace(etaini, etafin, 1001)
    Df0 = np.exp(etaini)
    Df_p0 = np.exp(etaini)
    y0 = np.array([Df0, Df_p0])

    if use_jax:
        # jax.experimental.ode.odeint expects signature func(y, t, *args)
        sol = jax_odeint(Deqs, y0, eta_arr)
        Dplus = sol[:, 0]
        Dplusp = sol[:, 1]
    else:
        # scipy.integrate.odeint returns array with shape (len(t), len(y))
        sol = scipy_odeint(lambda yy, tt: Deqs(yy, tt), y0, eta_arr)
        Dplus = sol[:, 0]
        Dplusp = sol[:, 1]

    # Interpolate to requested redshift
    Dplusp_ = interp(etaofz(zfin), eta_arr, Dplusp)
    Dplus_ = interp(etaofz(zfin), eta_arr, Dplus)
    f0 = Dplusp_/Dplus_

    return f0


def get_f0_jax_optimized(z, OmM0):
    """
    JAX-optimized version of get_f0 for high-performance computation.

    This function provides the same computation as get_f0 but is specifically
    optimized for JAX with JIT compilation, resulting in significant speedup.

    Args:
        z: redshift
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)

    Returns:
        f0: linear growth rate
    """
    # Only available when using JAX backend
    if backend != 'jax':
        raise RuntimeError("get_f0_jax_optimized requires JAX backend. "
                         "Use get_f0 for NumPy backend.")

    # Import JAX numpy for this function
    import jax.numpy as jnp
    from jax.experimental.ode import odeint

    def OmM(eta):
        return 1./(1. + ((1-OmM0)/OmM0)* jnp.exp(3*eta))

    def f1(eta):
        return 2. - 3./2. * OmM(eta)

    def f2(eta):
        return 3./2. * OmM(eta)

    etaini = -6  # initial eta, early enough to evolve as EdS (D+ ∝ a)
    zfin = z

    def etaofz(z):
        return jnp.log(1./(1. + z))

    etafin = etaofz(zfin)

    # Differential eqs.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return jnp.array([Dprime, f2(eta)*Df - f1(eta)*Dprime])

    # eta range and initial conditions
    eta_range = jnp.linspace(etaini, etafin, 1001)
    Df0 = jnp.exp(etaini)
    Df_p0 = jnp.exp(etaini)

    # solution
    solution = odeint(Deqs, jnp.array([Df0, Df_p0]), eta_range)
    Dplus = solution[:, 0]
    Dplusp = solution[:, 1]

    # Interpolate to requested redshift using JAX interp
    eta_target = etaofz(zfin)
    Dplusp_ = jnp.interp(eta_target, eta_range, Dplusp)
    Dplus_ = jnp.interp(eta_target, eta_range, Dplus)
    f0 = Dplusp_/Dplus_

    return f0


# JIT-compiled version for maximum performance
if backend == 'jax':
    import jax
    get_f0_jax_jit = jax.jit(get_f0_jax_optimized)
else:
    get_f0_jax_jit = None



# ── bias-monomial decomposition of the bispectrum ─────────────────────────────

class BiasPolynomial:
    r"""
    A polynomial in the bias parameters whose coefficients are arrays over the
    :math:`(k_1, k_2)` pairs and the angular quadrature.

    The redshift-space bispectrum is a polynomial in every bias parameter --
    :meth:`BispectrumCalculator.Z2` is linear in ``(b1, b2, bs)``, ``Z1eft`` is linear in
    ``(b1, c1, c2)``, the shot-noise term is linear in ``(Bshot, Pshot)`` plus one
    ``Pshot**2`` -- and the only non-polynomial dependence is the Finger-of-God kernel, which
    involves ``X_FoG`` alone.  Carrying the decomposition explicitly lets the whole
    bias-independent part (the AP transform, the power spectrum interpolations, the ``Z2``
    kernels and the angular quadrature) be evaluated once per cosmology instead of once per
    parameter point.

    A polynomial is a mapping from monomials to coefficient arrays, where a monomial is a
    sorted tuple of ``(bias name, exponent)`` pairs -- ``(('b1', 2), ('c2', 1))`` for
    :math:`b_1^2 c_2`, and ``()`` for the bias-independent term.  Naming the parameters in the
    monomial itself keeps each statistic to the parameters it actually uses and makes the tables
    self-describing: nothing downstream has to know a parameter ordering to read them.
    Arithmetic mirrors the expressions in :meth:`BispectrumCalculator.bispectrum`, so the
    monomial version can be read line by line against the direct one.

    Parameters
    ----------
    terms : dict, optional
        ``{monomial: coefficient}``, with *monomial* as above.
    """
    __slots__ = ('terms',)

    def __init__(self, terms=None):
        self.terms = dict(terms) if terms else {}

    @staticmethod
    def _product(left, right):
        """The monomial obtained by multiplying two monomials: exponents add, per name."""
        merged = dict(left)
        for name, exponent in right:
            merged[name] = merged.get(name, 0) + exponent
        return tuple(sorted(merged.items()))

    @classmethod
    def constant(cls, coefficient):
        """Return the polynomial whose only term is the bias-independent *coefficient*."""
        return cls({(): coefficient})

    @classmethod
    def variable(cls, name, coefficient=1.):
        """Return *coefficient* times the bias parameter *name*."""
        return cls({((name, 1),): coefficient})

    def __add__(self, other):
        if not isinstance(other, BiasPolynomial):
            other = BiasPolynomial.constant(other)
        terms = dict(self.terms)
        for exponents, coefficient in other.terms.items():
            terms[exponents] = terms[exponents] + coefficient if exponents in terms else coefficient
        return BiasPolynomial(terms)

    __radd__ = __add__

    def __neg__(self):
        return BiasPolynomial({exponents: -coefficient for exponents, coefficient in self.terms.items()})

    def __sub__(self, other):
        return self + (-other if isinstance(other, BiasPolynomial) else BiasPolynomial.constant(-other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, BiasPolynomial):  # scalar or array factor
            return BiasPolynomial({exponents: coefficient * other for exponents, coefficient in self.terms.items()})
        terms = {}
        for left_exponents, left in self.terms.items():
            for right_exponents, right in other.terms.items():
                exponents = self._product(left_exponents, right_exponents)
                product = left * right
                terms[exponents] = terms[exponents] + product if exponents in terms else product
        return BiasPolynomial(terms)

    __rmul__ = __mul__

    def monomials(self):
        """Return the monomials present, in a deterministic (sorted) order."""
        return sorted(self.terms)

    def stack(self, monomials, shape=None):
        """Return the coefficients for *monomials*, broadcast to a common shape and stacked.

        Missing monomials contribute zeros, so several polynomials can share one basis.  *shape*
        overrides the shape inferred from this polynomial's own terms, and is required when it
        has none -- which happens legitimately, e.g. the undamped block when ``damping_method``
        damps every term.
        """
        if shape is None:
            shape = np.broadcast_shapes(*[np.shape(coefficient) for coefficient in self.terms.values()])
        zero = np.zeros(shape)
        return np.array([np.broadcast_to(self.terms[exponents], shape) if exponents in self.terms else zero
                         for exponents in monomials])

    def substitute(self, values):
        """Return this polynomial with the bias parameters in *values* replaced by numbers.

        ``{'c2': 0.}`` collapses every monomial containing ``c2``, which is worth doing when a
        parameter is held fixed in the fit: it is the size of the monomial basis, not the number
        of bias parameters, that sets the table size.
        """
        terms = {}
        for exponents, coefficient in self.terms.items():
            reduced, dropped = [], False
            for name, exponent in exponents:
                if name not in values:
                    reduced.append((name, exponent))
                    continue
                value = values[name]
                if value == 0:
                    dropped = True
                    break
                coefficient = coefficient * value**exponent
            if dropped:
                continue
            key = tuple(reduced)
            terms[key] = terms[key] + coefficient if key in terms else coefficient
        return BiasPolynomial(terms)

    def evaluate(self, bias_values):
        """Evaluate at *bias_values*, a mapping of bias name to value (missing names are zero)."""
        total = 0.
        for exponents, coefficient in self.terms.items():
            for name, exponent in exponents:
                coefficient = coefficient * bias_values.get(name, 0.)**exponent
            total = total + coefficient
        return total


def bias_monomial_values(bias_values, monomials):
    """Return the monomial vector :math:`m_j` matching :meth:`BiasPolynomial.stack`'s *monomials*.

    Each monomial names the bias parameters it involves, so no shared ordering is needed;
    *bias_values* maps names to values and names it omits are taken as zero, which is what a
    monomial containing them contributes.
    """
    out = []
    for exponents in monomials:
        term = 1.
        for name, exponent in exponents:
            term = term * bias_values.get(name, 0.)**exponent
        out.append(term)
    return np.array(out)


def fog_lambda_nodes(k1k2pairs=None, f_max=1.2, n_nodes=24, dynamic_range=1e-5, lambda_max=None):
    r"""Return a fixed log-spaced grid of the FoG collocation variable
    :math:`\Lambda = \frac{f^2}{2}\sum_i (k_i \mu_i)^2`.

    Every damping kernel of :func:`fog_damping` depends on the angles **only** through
    :math:`\Lambda` (for ``'lor'`` and ``'exp'`` through :math:`X^2 \Lambda`, for ``'vdg'``
    through both :math:`\Lambda` and :math:`X^2\Lambda`), so interpolating the kernel on a grid
    in :math:`\Lambda` factorizes the ``X_FoG`` dependence out of the angular integral without
    approximating the ``X_FoG`` dependence itself.

    The grid is logarithmic because :math:`1/(1 + X^2\Lambda)` is a single fixed-shape logistic
    sigmoid in :math:`\log\Lambda`, merely translated by :math:`\log X^2`: one grid is then
    uniformly accurate over the whole ``X_FoG`` prior, which a grid in ``X_FoG`` would not be.

    The nodes must not depend on cosmology (they are part of the emulated table structure), so
    the range is set by the requested wavenumbers and a generous bound *f_max* on the growth
    rate; :math:`\Lambda` outside it is clamped to the end nodes.
    """
    import numpy as _np_native  # the node grid is fixed structure, never a traced array
    if lambda_max is None:
        k1k2 = _np_native.asarray(k1k2pairs)
        k1_max, k2_max = _np_native.max(k1k2[:, 0]), _np_native.max(k1k2[:, 1])
        k3_max = k1_max + k2_max  # |k3| <= k1 + k2, attained at x12 = 1
        lambda_max = 0.5 * f_max**2 * (k1_max**2 + k2_max**2 + k3_max**2)
    return _np_native.geomspace(lambda_max * dynamic_range, lambda_max, n_nodes)


def fog_collocation_weights(lam, lambda_nodes):
    """Return the interpolation weights of *lam* on *lambda_nodes*, shape ``lam.shape + (n_nodes,)``.

    Local cubic Lagrange interpolation in :math:`\\log\\Lambda` on the four nodes bracketing each
    point: fourth-order accurate in the node spacing, and the weights sum to one, so a kernel
    that is constant in :math:`\\Lambda` is reproduced exactly.
    """
    import numpy as _np_native  # the nodes are fixed structure, never a traced array
    log_nodes = _np_native.log(_np_native.asarray(lambda_nodes))
    n_nodes = len(log_nodes)
    if n_nodes < 4:
        raise ValueError(f'need at least 4 collocation nodes, got {n_nodes}')
    step = log_nodes[1] - log_nodes[0]
    position = (np.log(np.clip(lam, lambda_nodes[0], lambda_nodes[-1])) - log_nodes[0]) / step
    index = np.clip(np.floor(position) - 1, 0, n_nodes - 4)
    offset = position - (index + 1.)
    lagrange = [-offset * (offset - 1.) * (offset - 2.) / 6.,
                (offset + 1.) * (offset - 1.) * (offset - 2.) / 2.,
                -(offset + 1.) * offset * (offset - 2.) / 2.,
                (offset + 1.) * offset * (offset - 1.) / 6.]
    node_index = np.arange(n_nodes)
    basis = 0.
    for shift, weight in enumerate(lagrange):
        basis = basis + weight[..., None] * (node_index == (index + shift)[..., None])
    return basis


def fog_damping_correction(X_FoG, lambda_nodes, sigma2v=1., damping='lor', nlegs=3):
    r"""Return :math:`1 - W(X_\mathrm{FoG}, \Lambda_n)` at the collocation nodes.

    The damped block is reconstructed as ``damped - sum_n correction_n * tables_n``: writing the
    kernel as a *correction* to the undamped result rather than as the kernel itself makes the
    small-:math:`\Lambda` end harmless (the correction vanishes there, so the grid can be
    truncated) and makes ``X_FoG = 0`` exact by construction.
    """
    lam = np.asarray(lambda_nodes)
    lX2 = X_FoG**2 * lam
    if damping is None:
        return np.zeros_like(lam)
    if damping == 'lor':
        return 1. - 1. / (1. + lX2 * sigma2v)
    if damping == 'exp':
        return 1. - np.exp(-lX2 * sigma2v)
    if damping == 'vdg':
        denom = 1. + lX2
        return 1. - np.exp(-lam * sigma2v / denom) / denom**(nlegs - 1.5)
    raise ValueError(f"damping must be None, 'exp', 'lor' or 'vdg', got {damping!r}")




def fog_damping(*kmu_X, f=1., sigma2v=1., damping='lor'):
    r"""
    Finger-of-God damping kernel W, following jaxpower's pt.py convention.

    Parameters
    ----------
    kmu_X : tuples
        One ``(k * mu, X_FoG)`` pair per power spectrum leg: two (identical) pairs
        for the auto power spectrum, three for the bispectrum.
    f : float
        Growth rate :math:`f_0` (each ``k * mu`` is multiplied by ``f``).
    sigma2v : float
        Velocity dispersion :math:`\sigma_v^2`.
    damping : {None, 'exp', 'lor', 'vdg'}
        ``None`` returns 1 (no damping).

    Notes
    -----
    With :math:`\lambda_X^2 = \frac{f^2}{2} \sum_i (k_i \mu_i X_i)^2` and
    :math:`\lambda^2 = \frac{f^2}{2} \sum_i (k_i \mu_i)^2`:
    'exp' returns :math:`e^{-\lambda_X^2 \sigma_v^2}`, 'lor' returns
    :math:`1 / (1 + \lambda_X^2 \sigma_v^2)`, and 'vdg' returns
    :math:`e^{-\lambda^2 \sigma_v^2 / (1 + \lambda_X^2)} / (1 + \lambda_X^2)^{n - 3/2}`
    with :math:`n` the number of legs (1/2 for the power spectrum, 3/2 for the bispectrum).
    """
    if damping is None:
        return 1.
    lX2 = 0.5 * f**2 * sum((kmu * X)**2 for kmu, X in kmu_X)
    if damping == 'lor':
        return 1. / (1. + lX2 * sigma2v)
    if damping == 'exp':
        return np.exp(-lX2 * sigma2v)
    if damping == 'vdg':
        l2 = 0.5 * f**2 * sum(kmu**2 for kmu, _ in kmu_X)
        denom = 1. + lX2
        return np.exp(-l2 * sigma2v / denom) / denom**(len(kmu_X) - 1.5)
    raise ValueError(f"damping must be None, 'exp', 'lor' or 'vdg', got {damping!r}")


def _normalize_damping_method(damping_method):
    """Normalize / validate ``damping_method``; see :meth:`RSDMultipolesPowerSpectrumCalculator.get_rsd_pkmu`.

    ``None`` (the default) is an alias for ``'tree+loop+ctr'``, ``'all'`` for
    ``'tree+loop+ctr+sn'``; legacy ``'tree'`` and ``'tree-gtns'`` are deprecated and raise.
    """
    if damping_method is None:
        return 'tree+loop+ctr'
    if damping_method in ('tree', 'tree-gtns'):
        raise ValueError(f"damping_method={damping_method!r} is deprecated; use 'tree+loop+ctr' (GTNS removed)")
    if damping_method == 'all':
        return 'tree+loop+ctr+sn'
    if damping_method not in ('loop+ctr', 'tree+loop', 'tree+loop+ctr', 'tree+loop+ctr+sn'):
        raise ValueError(f"damping_method must be None / 'tree+loop+ctr' (default), 'tree+loop' or 'tree+loop+ctr+sn' (alias 'all'), got {damping_method!r}")
    return damping_method


def _resolve_use_gtns(use_GTNS, damping_method):
    """Resolve the GTNS switch to a bool; see :meth:`RSDMultipolesPowerSpectrumCalculator.get_eft_pkmu`.

    ``use_GTNS=None`` (the default) reproduces the ``damping_method``-driven behavior: GTNS is
    kept for ``'loop+ctr'`` (where the tree-level Kaiser term is undamped, so nothing resums
    it) and dropped for every ``'tree+...'`` method (where the tree-level damping resums it
    non-perturbatively, so keeping it would double count at O(lambda^2)).  ``True`` / ``False``
    force it on / off, whatever ``damping_method`` says.
    """
    if use_GTNS is None:
        return not damping_method.startswith('tree')
    if use_GTNS not in (True, False):
        raise ValueError(f'use_GTNS must be None, True or False, got {use_GTNS!r}')
    return bool(use_GTNS)




class BispectrumCalculator:
    def __init__(self, model='EFT'):
        """
        model : str
            'EFT', 'TNS', or 'FOLPSD'.
        """
        # self.basis = basis.lower()
        # if self.basis not in ['sugiyama', 'scoccimarro']:
        #     raise ValueError("basis must be 'sugiyama' or 'scoccimarro'.")
        self.model = model
        self._printed_model_damping_bk = True

    def set_bias_scheme(self, pars, bias_scheme="folps"):
        """Sets the nuisance parameters based on the selected bias scheme."""
        if bias_scheme in ["folps", "pat", "mcdonald","FOLPS","FolpsD","FOLPSD"]:
            if pars is None:
                pars = [1.0, 0.0, 0.0, 0.01, 0.01, 1, 1, 0]
            core, png, X_FoG_bk = split_png_pars(pars, 7, 3)
            (b1, b2, bs2, c1, c2, Bshot, Pshot) = core

        elif bias_scheme in ["Assassi", "classpt","assassi"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for Assassi/classpt bias scheme.")
            core, png, X_FoG_bk = split_png_pars(pars, 7, 3)
            (b1_classPT, b2_classPT, bG2_classPT, c1, c2, Bshot, Pshot) = core
            b1 = b1_classPT
            b2 = b2_classPT - 4/3 * bG2_classPT
            bs2 = 2 * bG2_classPT

        elif bias_scheme in ["DESI", "priordocument","DR2","priordoc"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for 'priordoc' bias scheme.")
            core, png, X_FoG_bk = split_png_pars(pars, 7, 3)
            (b1_priordoc, b2_priordoc, bK2_priordoc, c1, c2, Bshot, Pshot) = core
            b1 = b1_priordoc
            b2 = b2_priordoc
            bs2 = 2.* bK2_priordoc

        else:
            raise ValueError("Invalid bias scheme. Choose from 'folps' or 'classpt' or 'priordoc'.")

        # Always returned in the long, PNG-carrying convention, whatever the input length was.
        return [b1, b2, bs2, c1, c2, Bshot, Pshot, png[0], png[1], png[2], X_FoG_bk]

    #GL pairs [[x1,w1],[x2,w2],....
    _tables_cache = {}

    def _gauss_legendre(self, n, a=-1.0, b=1.0):
        """Return Gauss-Legendre nodes and weights on [a, b] for current backend."""
        # To remain JAX-jittable we always compute points on the host using NumPy,
        # then convert to the active backend array type exactly once.
        import numpy as _np  # type: ignore

        nodes, weights = _np.polynomial.legendre.leggauss(int(n))
        nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        weights = 0.5 * (b - a) * weights

        return np.asarray(nodes), np.asarray(weights)

    def tablesGL_f(self, precision=[4, 5, 5]):
        precision_key = tuple(int(p) for p in precision)
        cached = self._tables_cache.get(precision_key)
        if cached is not None:
            return cached

        Nphi, Nx, Nmu = precision_key

        phi_roots, phi_weights = self._gauss_legendre(Nphi, 0.0, np.pi)
        x_roots, x_weights = self._gauss_legendre(Nx, -1.0, 1.0)
        mu_roots, mu_weights = self._gauss_legendre(Nmu, -1.0, 1.0)

        phiGL = np.stack((phi_roots, phi_weights), axis=-1)
        xGL = np.stack((x_roots, x_weights), axis=-1)
        muGL = np.stack((mu_roots, mu_weights), axis=-1)

        tables = [phiGL, xGL, muGL]
        self._tables_cache[precision_key] = tables
        return tables

    def tablesGL2_f(self, precision=[10, 10]):   # For Scoccimarro

        precision_key = tuple(int(p) for p in precision)
        cached = self._tables_cache.get(precision_key)
        Nphi,Nmu = precision_key
        Pi= np.pi

        a, b = 0, np.pi
        phi_roots, phi_weights = scipy.special.roots_legendre(Nphi)
        phi_roots = 0.5 * (b - a) * phi_roots + 0.5 * (a + b)
        phi_weights = 0.5 * (b - a) * phi_weights
        phiGL=np.array([phi_roots,phi_weights]).T

        mu_roots, mu_weights = scipy.special.roots_legendre(Nmu)
        muGL=np.array([mu_roots,mu_weights]).T
        tablesGL = [phiGL,muGL]
        self._tables_cache[precision_key] = tablesGL

        return tablesGL


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
        x31AP = -(k1AP + k2AP*x12AP)/k3AP
        x23AP = -(k2AP + k1AP*x12AP)/k3AP

        return (k1AP, k2AP, k3AP,x12AP, x23AP, x31AP,mu1AP, mu2AP, mu3AP,cosphi)

    def Z2(self, ki, kj, xij, mui, muj, f, b1, b2, bs,A=1,Ap=0, bfnld=0., alphai=0., alphaj=0.):
        r"""Second-order kernel.  *bfnld* is :math:`b_{\phi\delta} f_\mathrm{NL}`, the
        second-order local-PNG bias, whose kernel is :math:`(\alpha_i + \alpha_j) / 2`."""

        term1 = b2/2 + bs/2 * (xij**2 - 1/3) + bfnld * (alphai + alphaj) / 2
        km=ki*mui + kj*muj
        term2 = km/2 * (mui/ki * f * (b1 + f * muj**2) +
                        muj/kj * f * (b1 + f * mui**2))
        F2 = 5/7 + xij/2 * (ki/kj + kj/ki) + 2/7 * xij**2
        G2 = 3/7 + xij/2 * (ki/kj + kj/ki) + 4/7 * xij**2
        term3 = b1 * F2
        mu2 = km**2 / (ki**2 + kj**2 + 2 * ki * kj * xij)
        term4 = f * mu2 * G2

        return term1 + term2 + term3 + term4



    def bispectrum(self, k1, k2, x12, mu1, phi, f, sigma2v, Sigma2, deltaSigma2,
                   bpars, qpar, qperp, k_pkl_pklnw, damping = 'lor',interpolation_method= 'cubic',
                   alphak2=None):

        core, (fnl, bphi, bphid), X_FoG_b = split_png_pars(bpars, 7, 3)
        b1, b2, bs, c1, c2, Bshot, Pshot = core

        cosphi = np.cos(phi)
        APtransf = self.APtransforms(k1, k2, x12, mu1, cosphi, qpar, qperp)
        k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP, cosphi = APtransf

        k_     = k_pkl_pklnw[0]
        pkl_   = k_pkl_pklnw[1]
        pklnw_ = k_pkl_pklnw[2]
        # alphak2, when given, is alpha(k) k^2 on the k_pkl_pklnw grid: the local-PNG
        # transfer, and what switches the PNG terms on.  See
        # :meth:`NonLinearPowerSpectrumCalculator._initialize_png` and :func:`table_alphak2`.
        png = alphak2 is not None

        interp_method = interpolation_method
        pk1   = self.interpolation_b(k1AP, k_, pkl_,   method=interp_method)
        pk1nw = self.interpolation_b(k1AP, k_, pklnw_, method=interp_method)
        pk2   = self.interpolation_b(k2AP, k_, pkl_,   method=interp_method)
        pk2nw = self.interpolation_b(k2AP, k_, pklnw_, method=interp_method)
        pk3   = self.interpolation_b(k3AP, k_, pkl_,   method=interp_method)
        pk3nw = self.interpolation_b(k3AP, k_, pklnw_, method=interp_method)

        e1IR = (1 + f*mu1AP**2 *(2 + f))*Sigma2 + (f*mu1AP)**2 * (mu1AP**2 - 1)* deltaSigma2
        e2IR = (1 + f*mu2AP**2 *(2 + f))*Sigma2 + (f*mu2AP)**2 * (mu2AP**2 - 1)* deltaSigma2
        e3IR = (1 + f*mu3AP**2 *(2 + f))*Sigma2 + (f*mu3AP)**2 * (mu3AP**2 - 1)* deltaSigma2

        pkIR1= pk1nw + (pk1-pk1nw)*np.exp(-e1IR*k1AP**2)
        pkIR2= pk2nw + (pk2-pk2nw)*np.exp(-e2IR*k2AP**2)
        pkIR3= pk3nw + (pk3-pk3nw)*np.exp(-e3IR*k3AP**2)


        f1 = f2 = f3 = f
        f0=f
        if png:
            alpha1 = self.interpolation_b(k1AP, k_, alphak2, method=interp_method) / k1AP**2
            alpha2 = self.interpolation_b(k2AP, k_, alphak2, method=interp_method) / k2AP**2
            alpha3 = self.interpolation_b(k3AP, k_, alphak2, method=interp_method) / k3AP**2
        else:
            alpha1 = alpha2 = alpha3 = 0.
        # Local-PNG scale-dependent bias, one per leg.
        bfnld = fnl * bphid
        Z1_1 = b1 + fnl * bphi * alpha1 + f1 * mu1AP**2
        Z1_2 = b1 + fnl * bphi * alpha2 + f2 * mu2AP**2
        Z1_3 = b1 + fnl * bphi * alpha3 + f3 * mu3AP**2

        Z1eft1 = Z1_1 - (c1 * mu1AP**2 + c2 * mu1AP**4) * k1AP**2
        Z1eft2 = Z1_2 - (c1 * mu2AP**2 + c2 * mu2AP**4) * k2AP**2
        Z1eft3 = Z1_3 - (c1 * mu3AP**2 + c2 * mu3AP**4) * k3AP**2


        B12 = (2 * self.Z2(k1AP, k2AP, x12AP, mu1AP, mu2AP, f, b1, b2, bs, bfnld=bfnld, alphai=alpha1, alphaj=alpha2) * Z1eft1*pkIR1 * Z1eft2*pkIR2)
        B23 = (2 * self.Z2(k2AP, k3AP, x23AP, mu2AP, mu3AP, f, b1, b2, bs, bfnld=bfnld, alphai=alpha2, alphaj=alpha3) * Z1eft2*pkIR2 * Z1eft3*pkIR3)
        B31 = (2 * self.Z2(k3AP, k1AP, x31AP, mu3AP, mu1AP, f, b1, b2, bs, bfnld=bfnld, alphai=alpha3, alphaj=alpha1) * Z1eft3*pkIR3 * Z1eft1*pkIR1)
        # Primordial bispectrum: B_m = M1 M2 M3 B_phi with B_phi the local template, propagated
        # through the three linear kernels.  Written with the alpha ratios so that it reuses the
        # IR-resummed spectra rather than a second set of interpolations.  alpha itself is the
        # linear-theory transfer: the exactly-resummed term would instead carry
        # sqrt(P_IR / P_L) per leg, a (1 - D) x w / 2 effect -- sub-percent of a term that is
        # itself small wherever the BAO wiggles are not, and the usual convention.
        Bprim = 0.
        if png:
            Bprim = 2. * fnl * Z1eft1 * Z1eft2 * Z1eft3 * (
                png_prim_ratio(alpha1, alpha2, alpha3) * pkIR1 * pkIR2
                + png_prim_ratio(alpha2, alpha3, alpha1) * pkIR2 * pkIR3
                + png_prim_ratio(alpha3, alpha1, alpha2) * pkIR3 * pkIR1)

        W = fog_damping((k1AP * mu1AP, X_FoG_b), (k2AP * mu2AP, X_FoG_b), (k3AP * mu3AP, X_FoG_b),
                        f=f, sigma2v=sigma2v, damping=damping)


        if not getattr(self, '_printed_model_damping_bk', False):
            print(f"[FOLPS] Model Bk: {self.model}, Damping: {damping}")
            self._printed_model_damping_bk = True


        # if self.model == "EFT":
        #     W = 1
        # elif self.model == "TNS":
        #     global use_TNS_model_status
        #     if use_TNS_model_status is not True:
        #         raise RuntimeError("[FOLPS] TNS: set use_TNS_model=True in MatrixCalculator.")
        #     if damping == 'lor':
        #         W = Wlor
        #     elif damping == 'vdg':
        #         W = Winfty
        #     elif damping == 'exp':
        #         W = Wexp
        #     else:
        #         W = Wlor
        # elif self.model == "FOLPSD":
        #     if damping not in ['lor', 'vdg','exp']:
        #         print("[FOLPS] For FOLPSD you must specify a valid damping ('lor', 'vdg','exp'). Default: 'lor'.")
        #         damping = 'lor'
        #     if damping == 'lor':
        #         W = Wlor
        #     elif damping == 'vdg':
        #         W = Winfty
        #     elif damping == 'exp':
        #         W = Wexp
        #     else:
        #         W = Wlor
        # else:
        #     if damping == 'lor':
        #         W = Wlor
        #     elif damping == 'vdg':
        #         W = Winfty
        #     elif damping == 'exp':
        #         W = Wexp
        #     else:
        #         W = Wlor

        ## Noise
        # To match eq.3.14 of 2110.10161, one makes (1+Pshot) -> (1+Pshot)/bar-n; Bshot -> Bshot/bar-n
        # The legs carry the PNG bias through Z1eft; the b1 multiplying Bshot does not, Bshot
        # being a free stochastic amplitude degenerate with it.  PNG stochastic terms of their
        # own (a <eps eps_phi> piece going as 1/alpha) are not modelled.
        shot = (
                  (b1*Bshot + 2.0*Pshot*f1*mu1AP**2) * Z1eft1 * pkIR1
                + (b1*Bshot + 2.0*Pshot*f2*mu2AP**2) * Z1eft2 * pkIR2
                + (b1*Bshot + 2.0*Pshot*f3*mu3AP**2) * Z1eft3 * pkIR3
                + Pshot**2
        )

        bispectrum = W*(B12 + B23 + B31 + Bprim) + shot
        alpha = qpar * qperp**2
        bispectrum = bispectrum / alpha**2

        return bispectrum

    def Z2_monomials(self, ki, kj, xij, mui, muj, f, alphai=None, alphaj=None):
        """:meth:`Z2` as a :class:`BiasPolynomial`; linear in ``b1``, ``b2`` and ``bs``, and in
        ``bphid`` times ``fnl`` when the local-PNG transfers *alphai*, *alphaj* are given."""
        km = ki * mui + kj * muj
        F2 = 5/7 + xij/2 * (ki/kj + kj/ki) + 2/7 * xij**2
        G2 = 3/7 + xij/2 * (ki/kj + kj/ki) + 4/7 * xij**2
        mu2 = km**2 / (ki**2 + kj**2 + 2 * ki * kj * xij)
        # term2 = km/2 * (mui/ki * f * (b1 + f muj**2) + muj/kj * f * (b1 + f mui**2)),
        # split into its b1-linear and bias-independent parts.
        term2_b1 = km / 2 * f * (mui / ki + muj / kj)
        term2_constant = km / 2 * f**2 * (mui / ki * muj**2 + muj / kj * mui**2)
        Z2 = (BiasPolynomial.variable('b2', 0.5)
              + BiasPolynomial.variable('bs', (xij**2 - 1/3) / 2)
              + BiasPolynomial.variable('b1', term2_b1 + F2)
              + BiasPolynomial.constant(term2_constant + f * mu2 * G2))
        if alphai is not None:
            Z2 = Z2 + (BiasPolynomial.variable('fnl')
                       * BiasPolynomial.variable('bphid', (alphai + alphaj) / 2))
        return Z2

    def bispectrum_monomials(self, k1, k2, x12, mu1, phi, f, sigma2v, Sigma2, deltaSigma2,
                             qpar, qperp, k_pkl_pklnw, interpolation_method='linear', alphak2=None):
        r""":meth:`bispectrum` decomposed over bias monomials.

        Returns ``(damped, undamped, lam)``: two :class:`BiasPolynomial` and the collocation
        variable :math:`\Lambda = \frac{f^2}{2}\sum_i (k_i\mu_i)^2`, such that for any bias
        vector and any Finger-of-God kernel

        ``bispectrum(bpars) == W(X_FoG, lam) * damped.evaluate(bpars) + undamped.evaluate(bpars)``

        with ``W`` the kernel of :func:`fog_damping`.  The shot-noise term is the undamped part.
        Line-by-line the same expressions as :meth:`bispectrum`, with the bias parameters
        carried symbolically instead of substituted.
        """
        cosphi = np.cos(phi)
        APtransf = self.APtransforms(k1, k2, x12, mu1, cosphi, qpar, qperp)
        k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP, cosphi = APtransf

        k_ = k_pkl_pklnw[0]
        pkl_ = k_pkl_pklnw[1]
        pklnw_ = k_pkl_pklnw[2]
        # alphak2, when given, switches the PNG terms on.  Absent, the monomial basis is
        # exactly the Gaussian one -- no zero-coefficient PNG monomials to carry around.
        png = alphak2 is not None

        interp_method = interpolation_method
        pk1 = self.interpolation_b(k1AP, k_, pkl_, method=interp_method)
        pk1nw = self.interpolation_b(k1AP, k_, pklnw_, method=interp_method)
        pk2 = self.interpolation_b(k2AP, k_, pkl_, method=interp_method)
        pk2nw = self.interpolation_b(k2AP, k_, pklnw_, method=interp_method)
        pk3 = self.interpolation_b(k3AP, k_, pkl_, method=interp_method)
        pk3nw = self.interpolation_b(k3AP, k_, pklnw_, method=interp_method)

        e1IR = (1 + f*mu1AP**2 * (2 + f))*Sigma2 + (f*mu1AP)**2 * (mu1AP**2 - 1) * deltaSigma2
        e2IR = (1 + f*mu2AP**2 * (2 + f))*Sigma2 + (f*mu2AP)**2 * (mu2AP**2 - 1) * deltaSigma2
        e3IR = (1 + f*mu3AP**2 * (2 + f))*Sigma2 + (f*mu3AP)**2 * (mu3AP**2 - 1) * deltaSigma2

        pkIR1 = pk1nw + (pk1 - pk1nw)*np.exp(-e1IR*k1AP**2)
        pkIR2 = pk2nw + (pk2 - pk2nw)*np.exp(-e2IR*k2AP**2)
        pkIR3 = pk3nw + (pk3 - pk3nw)*np.exp(-e3IR*k3AP**2)

        f1 = f2 = f3 = f

        if png:
            alpha1 = self.interpolation_b(k1AP, k_, alphak2, method=interp_method) / k1AP**2
            alpha2 = self.interpolation_b(k2AP, k_, alphak2, method=interp_method) / k2AP**2
            alpha3 = self.interpolation_b(k3AP, k_, alphak2, method=interp_method) / k3AP**2
        else:
            alpha1 = alpha2 = alpha3 = None

        def Z1eft(muAP, kAP, fi, alpha):
            """``b1 + b_phi fnl alpha + fi muAP**2 - (c1 muAP**2 + c2 muAP**4) kAP**2``."""
            Z1 = (BiasPolynomial.variable('b1')
                  + BiasPolynomial.constant(fi * muAP**2)
                  - BiasPolynomial.variable('c1', muAP**2 * kAP**2)
                  - BiasPolynomial.variable('c2', muAP**4 * kAP**2))
            if alpha is None:
                return Z1
            return Z1 + BiasPolynomial.variable('fnl') * BiasPolynomial.variable('bphi', alpha)

        Z1eft1 = Z1eft(mu1AP, k1AP, f1, alpha1)
        Z1eft2 = Z1eft(mu2AP, k2AP, f2, alpha2)
        Z1eft3 = Z1eft(mu3AP, k3AP, f3, alpha3)
        leg1, leg2, leg3 = Z1eft1 * pkIR1, Z1eft2 * pkIR2, Z1eft3 * pkIR3

        B12 = 2 * self.Z2_monomials(k1AP, k2AP, x12AP, mu1AP, mu2AP, f, alpha1, alpha2) * leg1 * leg2
        B23 = 2 * self.Z2_monomials(k2AP, k3AP, x23AP, mu2AP, mu3AP, f, alpha2, alpha3) * leg2 * leg3
        B31 = 2 * self.Z2_monomials(k3AP, k1AP, x31AP, mu3AP, mu1AP, f, alpha3, alpha1) * leg3 * leg1
        Bprim = BiasPolynomial()
        if png:
            # See :meth:`bispectrum`; the three legs are the same Z1eft polynomials.
            prim = (png_prim_ratio(alpha1, alpha2, alpha3) * pkIR1 * pkIR2
                    + png_prim_ratio(alpha2, alpha3, alpha1) * pkIR2 * pkIR3
                    + png_prim_ratio(alpha3, alpha1, alpha2) * pkIR3 * pkIR1)
            Bprim = (BiasPolynomial.variable('fnl', 2. * prim) * Z1eft1 * Z1eft2 * Z1eft3)

        Bshot, Pshot = BiasPolynomial.variable('Bshot'), BiasPolynomial.variable('Pshot')
        b1 = BiasPolynomial.variable('b1')
        shot = ((b1 * Bshot + Pshot * (2.0 * f1 * mu1AP**2)) * leg1
                + (b1 * Bshot + Pshot * (2.0 * f2 * mu2AP**2)) * leg2
                + (b1 * Bshot + Pshot * (2.0 * f3 * mu3AP**2)) * leg3
                + Pshot * Pshot)

        alpha = qpar * qperp**2
        lam = 0.5 * f**2 * ((k1AP * mu1AP)**2 + (k2AP * mu2AP)**2 + (k3AP * mu3AP)**2)
        return (B12 + B23 + B31 + Bprim) * (1. / alpha**2), shot * (1. / alpha**2), lam

    def Sugiyama_Bell_monomials(self, f, k_pkl_pklnw, k1k2pairs, qpar, qper,
                                precision=[8, 10, 10], multipoles=['B000', 'B202'],
                                renormalize=True, interpolation_method='linear',
                                lambda_nodes=None, n_lambda=12, fixed_bias=None, alphak2=None):
        r"""Angular-integrated bias-monomial tables for :meth:`Sugiyama_Bell`.

        Everything :meth:`Sugiyama_Bell` recomputes per parameter point that does not actually
        depend on the biases -- the AP transform, six power spectrum interpolations, the IR
        resummation, the ``Z2`` kernels and the 3-D angular quadrature -- is done once here, and
        the ``X_FoG`` dependence is factorized onto a collocation grid in
        :math:`\Lambda` (see :func:`fog_lambda_nodes`).  A parameter point then costs one
        contraction, :meth:`Sugiyama_Bell_from_monomials`.

        Returns
        -------
        dict
            ``'monomials'``: monomials as ``(bias name, exponent)`` pairs;
            ``'undamped'`` and ``'damped'``: ``(n_monomials, n_multipoles, n_pairs)``;
            ``'correction'``: ``(n_monomials, n_lambda, n_multipoles, n_pairs)``;
            ``'lambda_nodes'``, ``'sigma2v'``, ``'multipoles'``.
        """
        all_multipoles = ['B000', 'B110', 'B220', 'B202', 'B022', 'B112', 'B222']
        for mp in multipoles:
            if mp not in all_multipoles:
                raise ValueError(f"Invalid multipole '{mp}'. Available: {all_multipoles}")

        k1k2pairs = np.asarray(k1k2pairs)
        k1 = k1k2pairs[:, 0][:, None, None, None]
        k2 = k1k2pairs[:, 1][:, None, None, None]

        tablesGL = self.tablesGL_f(precision)
        sigma2v, Sigma2, deltaSigma2 = self.sigmas(k_pkl_pklnw[0], k_pkl_pklnw[1])

        phiGL, xGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        x, wx = xGL[:, 0], xGL[:, 1]
        mu, wmu = muGL[:, 0], muGL[:, 1]

        x_mesh = x[None, :, None, None]
        mu_mesh = mu[None, None, :, None]
        phi_mesh = phi[None, None, None, :]
        cosphi = np.cos(phi_mesh)
        cos2phi = np.cos(2 * phi_mesh)

        damped, undamped, lam = self.bispectrum_monomials(
            k1, k2, x_mesh, mu_mesh, phi_mesh, f, sigma2v, Sigma2, deltaSigma2,
            qpar, qper, k_pkl_pklnw, interpolation_method=interpolation_method, alphak2=alphak2)
        if fixed_bias:
            damped, undamped = damped.substitute(fixed_bias), undamped.substitute(fixed_bias)

        if lambda_nodes is None:
            lambda_nodes = fog_lambda_nodes(k1k2pairs, n_nodes=n_lambda)
        basis = fog_collocation_weights(lam, lambda_nodes)  # (..., n_lambda)

        # One monomial basis shared by both polynomials, so the two tables can be added.
        monomials = sorted(set(damped.monomials()) | set(undamped.monomials()))
        damped_coeffs = damped.stack(monomials)      # (n_monomials, n_pairs, n_x, n_mu, n_phi)
        undamped_coeffs = undamped.stack(monomials)

        Hl1l2L_dict = {'B000': 1.0, 'B110': -1 / np.sqrt(3), 'B220': 1 / np.sqrt(5),
                       'B202': 1 / np.sqrt(5), 'B022': 1 / np.sqrt(5), 'B112': np.sqrt(2 / 15),
                       'B222': -2 / np.sqrt(70)}

        damped_out, undamped_out, correction_out = [], [], []
        for mp in multipoles:
            integrand = self._compute_single_integrand(mp, x_mesh, mu_mesh, phi_mesh, cosphi, cos2phi)
            # The angular integration of Sugiyama_Bl1l2L, as a single set of quadrature weights.
            weight = 2 * integrand * wphi[None, None, None, :] * wmu[None, None, :, None] * wx[None, :, None, None]
            if renormalize:
                weight = weight * Hl1l2L_dict[mp]
            damped_out.append(np.sum(damped_coeffs * weight, axis=(-3, -2, -1)))
            undamped_out.append(np.sum(undamped_coeffs * weight, axis=(-3, -2, -1)))
            correction_out.append(np.einsum('jpxyz,pxyzn->jnp', damped_coeffs * weight, basis))

        return {'monomials': monomials,
                'fixed_bias': dict(fixed_bias or {}),
                'multipoles': list(multipoles),
                'lambda_nodes': np.asarray(lambda_nodes),
                'sigma2v': sigma2v,
                'damped': np.stack(damped_out, axis=1),
                'undamped': np.stack(undamped_out, axis=1),
                'correction': np.stack(correction_out, axis=2)}

    def Sugiyama_Bell_from_monomials(self, tables, bpars, damping='lor'):
        """Contract monomial *tables* from :meth:`BispectrumCalculator.Sugiyama_Bell_monomials`
        with the bias parameters, returning the tables' trailing axes -- ``(n_multipoles, n_pairs)``
        as built, or whatever a linear map has since contracted them onto.

        *bpars* is ``[b1, b2, bs, c1, c2, Bshot, Pshot, (fnl, bphi, bphid,) X_FoG]`` in the
        folps convention, the PNG entries optional.
        """
        # set_bias_scheme returns the folps convention, [b1, b2, bs, c1, c2, Bshot, Pshot,
        # (fnl, bphi, bphid,) X_FoG]; X_FoG is the one parameter the decomposition does not
        # absorb, and it enters through the damping correction below instead.
        names = ('b1', 'b2', 'bs', 'c1', 'c2', 'Bshot', 'Pshot', 'fnl', 'bphi', 'bphid')
        core, png, X_FoG = split_png_pars(bpars, 7, 3)
        monomial_values = bias_monomial_values(dict(zip(names, list(core) + list(png))), tables['monomials'])
        correction = fog_damping_correction(X_FoG, tables['lambda_nodes'], sigma2v=tables['sigma2v'],
                                            damping=damping)
        # Only two axes are named: the monomial axis (first) and, on the correction table, the
        # collocation axis (second).  Everything after them is left alone, so the same contraction
        # works whether the trailing axes are (n_multipoles, n_pairs) or the data bins a window
        # matrix has already been contracted onto.
        damped = tables['damped'] - np.tensordot(correction, tables['correction'], axes=([0], [1]))
        return np.tensordot(monomial_values, tables['undamped'] + damped, axes=([0], [0]))

    def interpolation_b(self, k_out, k_in, pk_in, method='cubic'):
        """Interpolation function

        Parameters:
        -----------
        k_out : array-like
            Output k values where interpolation is desired
        k_in : array-like
            Input k values
        pk_in : array-like
            Input power spectrum values
        method : str
            Interpolation method: 'linear' or 'cubic' (default)

        Returns:
        --------
        pk_out : array-like
            Interpolated power spectrum values
        """
        if method == 'linear':
            pk_out = np.interp(k_out, k_in, pk_in)
        elif method == 'cubic':
            if backend == 'jax':
                # Use interpax cubic2 which matches SciPy CubicSpline exactly
                pk_out = interp(k_out, k_in, pk_in, method='cubic2')
            else:
                # NumPy backend: use SciPy CubicSpline with not-a-knot boundary conditions
                from scipy.interpolate import CubicSpline
                spline = CubicSpline(k_in, pk_in, bc_type='not-a-knot', extrapolate=True)
                pk_out = spline(k_out)
        else:
            # Default to linear if method not recognized
            pk_out = np.interp(k_out, k_in, pk_in)

        return pk_out

    def sigmas(self, kT,pklT):

        k_BAO = 1/104
        kS = 0.4

        sigma2v_  = simpson(pklT, x=kT) / (6 * np.pi**2)
        sigma2v_ *= 1.05  #correction due to k cut

        # pklT_=pklT[kT<=0.4].copy()
        # kT_=kT[kT<=0.4].copy()

        mask = kT <= 0.4
        pklT_= np.where(mask, pklT, 0.0)
        kT_   = np.where(mask, kT, 0.0)

        j0_small = spherical_jn_backend(0, kT_ / k_BAO)
        j2_small = spherical_jn_backend(2, kT_ / k_BAO)
        Sigma2_ = 1/(6 * np.pi**2)*simpson(pklT_*(1 - j0_small + 2*j2_small), x=kT_)
        j2_full = spherical_jn_backend(2, kT / k_BAO)
        deltaSigma2_ = 1/(2 * np.pi**2)*simpson(pklT*j2_full, x=kT)

        return sigma2v_, Sigma2_, deltaSigma2_


    def angdep_integrands(self, x, mu, phi, cosphi, cos2phi):
        Pi = np.pi

        b000 = 1.0 / (8 * Pi)
        b110 = (-3 * np.sqrt(3) * x) / (8 * Pi)
        b220 = 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * x**2)
        b202 = 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * mu**2)

        b022 = (
            5 * np.sqrt(5) * (
                (-1 + 3 * mu**2) * (-1 + 3 * x**2)
                + 12 * mu * np.sqrt(1 - mu**2)
                * x * np.sqrt(1 - x**2) * cosphi
                + 3 * (-1 + mu**2) * (-1 + x**2) * cos2phi
            )
        ) / (32 * Pi)

        b112 = (
            3 * np.sqrt(2.5) * (
                np.sqrt(3) * (-1 + 3 * mu**2) * x
                + 3 * np.sqrt(3) * mu * np.sqrt(1 - mu**2)
                * np.sqrt(1 - x**2) * cosphi
            )
        ) / (8 * Pi)

        b222 = (
            25 * np.sqrt(70) * (
                1 - 3 * mu**2 * x**2
                - 3 * mu * x * np.sqrt(1 - mu**2)
                * np.sqrt(1 - x**2) * cosphi
                - 1.5 * (1 - mu**2) * (1 - x**2) * (1 - cos2phi)
            )
        ) / (112 * Pi)

        return b000, b110, b220, b202, b022, b112, b222

    def _compute_single_integrand(self, multipole, x, mu, phi, cosphi, cos2phi):
        """
        Compute angular dependence integrand for a single multipole.
        This saves computation time by only calculating what's needed.

        Parameters:
        -----------
        multipole : str
            The multipole to compute: 'B000', 'B110', 'B220', 'B202', 'B022', 'B112'
        x, mu, phi : array-like
            Angular coordinates
        cosphi, cos2phi : array-like
            Pre-computed cos(phi) and cos(2*phi)

        Returns:
        --------
        integrand : array
            The angular integrand for the requested multipole
        """
        Pi = np.pi

        if multipole == 'B000':
            return 1.0 / (8 * Pi)

        elif multipole == 'B110':
            return (-3 * np.sqrt(3) * x) / (8 * Pi)

        elif multipole == 'B220':
            return 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * x**2)

        elif multipole == 'B202':
            return 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * mu**2)

        elif multipole == 'B022':
            return (
                5 * np.sqrt(5) * (
                    (-1 + 3 * mu**2) * (-1 + 3 * x**2)
                    + 12 * mu * np.sqrt(1 - mu**2)
                    * x * np.sqrt(1 - x**2) * cosphi
                    + 3 * (-1 + mu**2) * (-1 + x**2) * cos2phi
                )
            ) / (32 * Pi)

        elif multipole == 'B112':
            # The cos(phi) coefficient is 3*sqrt(3), not 6: with mu2 = mu*x + sqrt(1-mu^2)
            # sqrt(1-x^2) cos(phi) the bracket is sqrt(3) * (3 mu mu2 - x), which is the
            # only form symmetric under k1 <-> k2, as (1 1 L) with even L must be.
            return (
                3 * np.sqrt(2.5) * (
                    np.sqrt(3) * (-1 + 3 * mu**2) * x
                    + 3 * np.sqrt(3) * mu * np.sqrt(1 - mu**2)
                    * np.sqrt(1 - x**2) * cosphi
                )
            ) / (8 * Pi)

        elif multipole == 'B222':
            # 125/(8 pi) * S_222; equals 25 sqrt(70)/(112 pi) *
            # [1 - 3 mu^2 x^2 - 3 mu x sqrt(1-mu^2) sqrt(1-x^2) cos(phi)
            #    - (3/2) (1-mu^2)(1-x^2) (1 - cos(2 phi))]
            return (
                25 * np.sqrt(70) * (
                    1 - 3 * mu**2 * x**2
                    - 3 * mu * x * np.sqrt(1 - mu**2)
                    * np.sqrt(1 - x**2) * cosphi
                    - 1.5 * (1 - mu**2) * (1 - x**2) * (1 - cos2phi)
                )
            ) / (112 * Pi)

        else:
            raise ValueError(f"Unknown multipole: {multipole}")


    def Sugiyama_Bl1l2L(self, k1, k2, f, sigma2v, Sigma2, deltaSigma2, bpars, qpar, qper,
                        tablesGL, k_pkl_pklnw, damping='lor', renormalize=True,
                        multipoles=['B000', 'B202'], interpolation_method='linear',precision=None,
                        alphak2=None):
        """
        Compute requested Sugiyama bispectrum multipoles.

        Parameters:
        -----------
        k1, k2 : float or array
            Wavenumbers
        f : float
            Growth rate
        sigma2v, Sigma2, deltaSigma2 : float
            IR resummation scales
        bpars : array-like
            Bias parameters [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b]
        qpar, qper : float
            AP parameters
        tablesGL : list
            Gauss-Legendre tables
        k_pkl_pklnw : list
            [k_array, pkl_array, pklnw_array]
        alphak2 : array, default=None
            alpha(k) k^2 on that same grid, switching the local-PNG terms on;
            see :func:`table_alphak2`.
        damping : str
            Damping type: 'lor', 'exp', 'vdg'
        renormalize : bool
            If True, apply Hl1l2L normalization factors
        multipoles : list of str
            List of multipoles to compute.
            Available: ['B000', 'B110', 'B220', 'B202', 'B022', 'B112']
        interpolation_method : str
            Interpolation method for power spectrum

        Returns:
        --------
        dict : Dictionary with requested multipoles as keys
        """
        Pi = np.pi

        phiGL, xGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        x, wx     = xGL[:, 0],  xGL[:, 1]
        mu, wmu   = muGL[:, 0], muGL[:, 1]

        x_mesh   = x[None, :, None, None]
        mu_mesh  = mu[None, None, :, None]
        phi_mesh = phi[None, None, None, :]

        cosphi  = np.cos(phi_mesh)
        cos2phi = np.cos(2 * phi_mesh)

        bisp = self.bispectrum(
            k1, k2, x_mesh, mu_mesh, phi_mesh,
            f, sigma2v, Sigma2, deltaSigma2,
            bpars, qpar, qper,
            k_pkl_pklnw,
            damping=damping,
            interpolation_method=interpolation_method,
            alphak2=alphak2,
        )

        # Normalization factors for each multipole
        Hl1l2L_dict = {
            'B000': 1.0,
            'B110': -1 / np.sqrt(3),
            'B220': 1 / np.sqrt(5),
            'B202': 1 / np.sqrt(5),
            'B022': 1 / np.sqrt(5),
            'B112': np.sqrt(2 / 15),
            'B222': -2 / np.sqrt(70)
        }

        # Compute only the requested multipoles (saves computation time)
        result = {}
        for mp in multipoles:
            # Calculate integrand only for requested multipole
            integrand = self._compute_single_integrand(mp, x_mesh, mu_mesh, phi_mesh, cosphi, cos2phi)

            # Perform angular integrations
            int_phi = 2 * np.sum(bisp * integrand * wphi[None,None,None,:], axis=3)
            int_mu  = np.sum(int_phi * wmu[None,None,:], axis=2)
            int_x   = np.sum(int_mu * wx[None,:], axis=1)

            if renormalize:
                int_x *= Hl1l2L_dict[mp]

            # Return scalar if k1, k2 are scalars
            if np.ndim(k1) == 0 and np.ndim(k2) == 0:
                result[mp] = float(int_x)
            else:
                result[mp] = int_x

        return result


    def Sugiyama_Bell(self, f, bpars, k_pkl_pklnw,
                      k1k2pairs, qpar, qper, precision=[8, 10, 10], damping='lor',
                      m_bin=None, k_thy=None, do_binning=False, multipoles=['B000', 'B202'],
                      renormalize=True, interpolation_method='linear', bias_scheme='folps',do_interp_bk=False,kout=None,
                      alphak2=None):
        """
        Compute Sugiyama bispectrum multipoles for multiple k1k2 pairs.

        Parameters:
        -----------
        f : float
            Growth rate
        bpars : array-like
            Bias parameters [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b]
        k_pkl_pklnw : list
            [k_array, pkl_array, pklnw_array]
        alphak2 : array, default=None
            alpha(k) k^2 on that same grid, switching the local-PNG terms on;
            see :func:`table_alphak2`.
        k1k2pairs : array-like
            Array of shape (N, 2) with k1, k2 pairs
        qpar, qper : float
            AP parameters
        precision : list
            Gauss-Legendre precision [Nphi, Nx, Nmu]
        damping : str
            Damping type: 'lor', 'exp', 'vdg'
        m_bin : array-like, optional
            Binning matrix (n_bins, n_theory)
        k_thy : array-like, optional
            Theory k values for binning
        do_binning : bool
            If True, perform internal binning
        multipoles : list of str
            Multipoles to compute. Default: ['B000', 'B202']
        renormalize : bool
            If True, apply normalization factors
        interpolation_method : str
            Interpolation method for power spectrum
        bias_scheme : str
            Bias scheme. Default: 'folps' (mcdonald basis)

        Returns:
        --------
        tuple : Tuple of arrays, one for each requested multipole
                If do_binning=True, arrays have shape (n_bins,)
                Otherwise, arrays have shape (N,) where N = len(k1k2pairs)
        """
        # Validate requested multipoles
        all_multipoles = ['B000', 'B110', 'B220', 'B202', 'B022', 'B112', 'B222']
        for mp in multipoles:
            if mp not in all_multipoles:
                raise ValueError(f"Invalid multipole '{mp}'. Available: {all_multipoles}")

        bpars = self.set_bias_scheme(bpars, bias_scheme=bias_scheme)
        Pi = np.pi

        # Convert k1k2pairs to array and extract k1, k2 with proper broadcasting dimensions
        k1k2pairs = np.asarray(k1k2pairs)
        k1 = k1k2pairs[:, 0][:, None, None, None]
        k2 = k1k2pairs[:, 1][:, None, None, None]

        # Tables for GL quadrature
        tablesGL = self.tablesGL_f(precision)

        # IR resummation scales
        sigma2v, Sigma2, deltaSigma2 = self.sigmas(k_pkl_pklnw[0], k_pkl_pklnw[1])

        phiGL, xGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        x, wx     = xGL[:, 0],  xGL[:, 1]
        mu, wmu   = muGL[:, 0], muGL[:, 1]

        x_mesh   = x[None, :, None, None]
        mu_mesh  = mu[None, None, :, None]
        phi_mesh = phi[None, None, None, :]

        cosphi  = np.cos(phi_mesh)
        cos2phi = np.cos(2 * phi_mesh)

        # Compute bispectrum for all k1k2 pairs at once (vectorized)
        bisp = self.bispectrum(
            k1, k2, x_mesh, mu_mesh, phi_mesh,
            f, sigma2v, Sigma2, deltaSigma2,
            bpars, qpar, qper,
            k_pkl_pklnw,
            damping=damping,
            interpolation_method=interpolation_method,
            alphak2=alphak2,
        )

        # Normalization factors for each multipole
        Hl1l2L_dict = {
            'B000': 1.0,
            'B110': -1 / np.sqrt(3),
            'B220': 1 / np.sqrt(5),
            'B202': 1 / np.sqrt(5),
            'B022': 1 / np.sqrt(5),
            'B112': np.sqrt(2 / 15),
            'B222': -2 / np.sqrt(70)
        }

        # Compute only the requested multipoles (saves computation time)
        multipoles_dict = {}
        for mp in multipoles:
            # Calculate integrand only for requested multipole
            integrand = self._compute_single_integrand(mp, x_mesh, mu_mesh, phi_mesh, cosphi, cos2phi)

            # Perform angular integrations
            int_phi = 2 * np.sum(bisp * integrand * wphi[None,None,None,:], axis=3)
            int_mu  = np.sum(int_phi * wmu[None,None,:], axis=2)
            int_x   = np.sum(int_mu * wx[None,:], axis=1)

            if renormalize:
                int_x *= Hl1l2L_dict[mp]

            multipoles_dict[mp] = int_x

        # Optional internal binning
        if do_binning:
            if m_bin is None or k_thy is None:
                raise ValueError("do_binning=True requires m_bin and k_thy to be provided.")
            m_bin_arr = np.asarray(m_bin)
            k_thy_arr = np.asarray(k_thy)
            x = np.asarray(k1k2pairs)[:, 0]

            binned_results = []
            for mp in multipoles:
                mp_interp = interp(k_thy_arr, x, multipoles_dict[mp])
                mp_binned = m_bin_arr @ mp_interp
                binned_results.append(mp_binned)

            return tuple(binned_results)

        # Optional internal interpolation to required k values: interpolate to theory k grid

        if do_interp_bk:
            if kout is None:
                raise ValueError("do_interp=True requires kout to be provided.")
            k_thy_arr = np.asarray(kout)
            x = np.asarray(k1k2pairs)[:, 0]
            interp_results = []
            for mp in multipoles:
                mp_interp = interp(k_thy_arr, x, multipoles_dict[mp])
                interp_results.append(mp_interp)

            return tuple(interp_results)

        # Return only requested multipoles
        return tuple(multipoles_dict[mp] for mp in multipoles)


    def Scoccimarro_B024(
        self,
        k1k2k3triplets,
        f,
        sigma2v,
        Sigma2,
        deltaSigma2,
        bpars,
        qpar,
        qperp,
        tablesGL,
        k_pkl_pklnw,
        damping = 'lor',
        interpolation_method='linear',
        multipoles=['B0', 'B2', 'B4'],
        alphak2=None
    ):
        """
        Computation of Scoccimarro B0, B2, B4
        Only computes the multipoles specified in the 'multipoles' parameter.
        """

        twopi = 2.0 * np.pi
        normB0, normB2, normB4 = 0.5, 2.5, 4.5


        k1k2k3triplets = np.asarray(k1k2k3triplets)
        k1 = k1k2k3triplets[:, 0][:, None, None]
        k2 = k1k2k3triplets[:, 1][:, None, None]
        k3 = k1k2k3triplets[:, 2][:, None, None]


        x = (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)

        # --- Gauss Legendre quads ---
        phiGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        mu,  wmu  = muGL[:, 0],  muGL[:, 1]

        mu_mesh  = mu[None, :, None]     # (1, Nμ, 1)
        phi_mesh = phi[None, None, :]    # (1, 1, Nφ)


        bisp = self.bispectrum(
            k1, k2,
            x,
            mu_mesh,
            phi_mesh,
            f,
            sigma2v,
            Sigma2,
            deltaSigma2,
            bpars,
            qpar,
            qperp,
            k_pkl_pklnw,
            damping = 'lor',
            interpolation_method = interpolation_method,
            alphak2=alphak2,
        )   # (N, Nμ, Nφ)

        # --- φ integration ---
        int_phi = 2.0 * np.sum(bisp * wphi[None, None, :], axis=2)  # (N, Nμ)

        # --- μ integration - Only compute requested multipoles ---
        results = {}

        if 'B0' in multipoles:
            B0 = np.sum(int_phi * wmu[None, :], axis=1) / twopi
            B0 *= normB0
            results['B0'] = B0
        else:
            results['B0'] = None

        if 'B2' in multipoles:
            leg2 = 0.5 * (-1.0 + 3.0 * mu**2)
            B2 = np.sum(int_phi * leg2[None, :] * wmu[None, :], axis=1) / twopi
            B2 *= normB2
            results['B2'] = B2
        else:
            results['B2'] = None

        if 'B4' in multipoles:
            leg4 = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
            B4 = np.sum(int_phi * leg4[None, :] * wmu[None, :], axis=1) / twopi
            B4 *= normB4
            results['B4'] = B4
        else:
            results['B4'] = None

        return results['B0'], results['B2'], results['B4'], x[:, 0, 0]



    def Scoccimarro_Bell(self, k1k2k3triplets, f, bpars, qpar, qperp,
                         k_pkl_pklnw, precision=[10, 10], damping='lor',
                         interpolation_method='linear', m_bin=None, k_thy=None,
                         do_binning=False, multipoles=['B0', 'B2', 'B4'], bias_scheme='folps',
                         alphak2=None):
        """
        Compute Scoccimarro bispectrum multipoles for multiple k1k2k3 triplets.

        Note: Normalization factors (0.5, 2.5, 4.5) are always applied to B0, B2, B4.

        Parameters:
        -----------
        k1k2k3triplets : array-like
            Array of shape (N, 3) with k1, k2, k3 triplets
        f : float
            Growth rate
        bpars : array-like
            Bias parameters [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b]
        qpar, qperp : float
            AP parameters
        k_pkl_pklnw : list
            [k_array, pkl_array, pklnw_array]
        alphak2 : array, default=None
            alpha(k) k^2 on that same grid, switching the local-PNG terms on;
            see :func:`table_alphak2`.
        precision : list
            Gauss-Legendre precision [Nphi, Nmu]
        damping : str
            Damping type: 'lor', 'exp', 'vdg'
        interpolation_method : str
            Interpolation method for power spectrum
        m_bin : array-like, optional
            Binning matrix (n_bins, n_theory)
        k_thy : array-like, optional
            Theory k values for binning
        do_binning : bool
            If True, perform internal binning
        multipoles : list of str
            Multipoles to compute. Default: ['B0', 'B2', 'B4']
        bias_scheme : str
            Bias scheme. Default: 'folps' (mcdonald basis)

        Returns:
        --------
        tuple : (B0, B2, B4, x) if not binning
                (B0_binned, B2_binned, B4_binned) if binning
        """
        bpars = self.set_bias_scheme(bpars, bias_scheme=bias_scheme)

        # IR resummation scales
        kT, pklT = k_pkl_pklnw[0], k_pkl_pklnw[1]
        sigma2v, Sigma2, deltaSigma2 = self.sigmas(kT, pklT)

        # GL tables for (phi, mu)
        tablesGL = self.tablesGL2_f(precision)

        # Fast path: if running with JAX, try to use jax.vmap to compute all
        # triplets at once (avoids Python loops). If that fails, fall back to a
        # list comprehension.
        if backend == 'jax':
            try:
                import jax

                # Build vmapped function that maps over three inputs (k1, k2, k3).
                def _single(k1, k2, k3):
                    # Create a single triplet array
                    triplet = np.array([[k1, k2, k3]])
                    return self.Scoccimarro_B024(
                        triplet, f, sigma2v, Sigma2, deltaSigma2, bpars, qpar, qperp,
                        tablesGL, k_pkl_pklnw, damping=damping,
                        interpolation_method=interpolation_method,
                        multipoles=multipoles, alphak2=alphak2
                    )

                vm = jax.vmap(_single)
                # k1k2k3triplets may be a Python list; convert to backend arrays first
                triplets_arr = np.asarray(k1k2k3triplets)
                k1s = triplets_arr[:, 0]
                k2s = triplets_arr[:, 1]
                k3s = triplets_arr[:, 2]
                stacked = vm(k1s, k2s, k3s)

                # stacked returns tuple (B0, B2, B4, x) for each triplet
                # Each element has shape (N,) with one extra dim from _single
                B0 = stacked[0][:, 0]  # Remove extra dim
                B2 = stacked[1][:, 0]
                B4 = stacked[2][:, 0]
                x = stacked[3][:, 0]

                # Optional internal binning
                if do_binning:
                    if m_bin is None or k_thy is None:
                        raise ValueError("do_binning=True requires m_bin and k_thy to be provided.")
                    m_bin_arr = np.asarray(m_bin)
                    k_thy_arr = np.asarray(k_thy)

                    # Use k1 values for binning
                    k_vals = triplets_arr[:, 0]

                    # Bin each multipole
                    B0_interp = interp(k_thy_arr, k_vals, B0)
                    B0_binned = m_bin_arr @ B0_interp

                    B2_interp = interp(k_thy_arr, k_vals, B2)
                    B2_binned = m_bin_arr @ B2_interp

                    B4_interp = interp(k_thy_arr, k_vals, B4)
                    B4_binned = m_bin_arr @ B4_interp

                    return B0_binned, B2_binned, B4_binned

                return B0, B2, B4, x

            except Exception:
                # If JAX vmapping isn't possible, fall back to list comprehension.
                pass

        # Fallback: call Scoccimarro_B024 with all triplets at once
        B0, B2, B4, x = self.Scoccimarro_B024(
            k1k2k3triplets, f, sigma2v, Sigma2, deltaSigma2, bpars, qpar, qperp,
            tablesGL, k_pkl_pklnw, damping, interpolation_method,
            multipoles=multipoles, alphak2=alphak2
        )

        # Optional internal binning
        if do_binning:
            if m_bin is None or k_thy is None:
                raise ValueError("do_binning=True requires m_bin and k_thy to be provided.")
            m_bin_arr = np.asarray(m_bin)
            k_thy_arr = np.asarray(k_thy)

            # Use k1 values for binning
            triplets_arr = np.asarray(k1k2k3triplets)
            k_vals = triplets_arr[:, 0]

            # Bin each multipole
            B0_interp = interp(k_thy_arr, k_vals, B0)
            B0_binned = m_bin_arr @ B0_interp

            B2_interp = interp(k_thy_arr, k_vals, B2)
            B2_binned = m_bin_arr @ B2_interp

            B4_interp = interp(k_thy_arr, k_vals, B4)
            B4_binned = m_bin_arr @ B4_interp

            return B0_binned, B2_binned, B4_binned

        return B0, B2, B4, x





class BispectrumCalculator_fk:
    def __init__(self, model='EFT'):
        """
        model : str
            'EFT', 'TNS', or 'FOLPSD'.
        """
        # self.basis = basis.lower()
        # if self.basis not in ['sugiyama', 'scoccimarro']:
        #     raise ValueError("basis must be 'sugiyama' or 'scoccimarro'.")
        self.model = model
        self._printed_model_damping_bk = True


    def set_bias_scheme(self, pars, bias_scheme="folps"):
        """Sets the nuisance parameters based on the selected bias scheme."""
        if bias_scheme in ["folps", "pat", "mcdonald","FOLPS","FolpsD","FOLPSD"]:
            if pars is None:
                pars = [1.0, 0.0, 0.0, 0.01, 0.01, 1, 1, 0]
            core, png, X_FoG_bk = split_png_pars(pars, 7, 3)
            (b1, b2, bs2, c1, c2, Bshot, Pshot) = core

        elif bias_scheme in ["Assassi", "classpt","assassi"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for Assassi/classpt bias scheme.")
            core, png, X_FoG_bk = split_png_pars(pars, 7, 3)
            (b1_classPT, b2_classPT, bG2_classPT, c1, c2, Bshot, Pshot) = core
            b1 = b1_classPT
            b2 = b2_classPT - 4/3 * bG2_classPT
            bs2 = 2 * bG2_classPT

        elif bias_scheme in ["DESI", "priordocument","DR2","priordoc"]:
            if pars is None:
                raise ValueError("Nuisance parameters must be provided for 'priordoc' bias scheme.")
            core, png, X_FoG_bk = split_png_pars(pars, 7, 3)
            (b1_priordoc, b2_priordoc, bK2_priordoc, c1, c2, Bshot, Pshot) = core
            b1 = b1_priordoc
            b2 = b2_priordoc
            bs2 = 2.* bK2_priordoc

        else:
            raise ValueError("Invalid bias scheme. Choose from 'folps' or 'classpt' or 'priordoc'.")

        # Always returned in the long, PNG-carrying convention, whatever the input length was.
        return [b1, b2, bs2, c1, c2, Bshot, Pshot, png[0], png[1], png[2], X_FoG_bk]

    #GL pairs [[x1,w1],[x2,w2],....
    _tables_cache = {}

    def _gauss_legendre(self, n, a=-1.0, b=1.0):
        """Return Gauss-Legendre nodes and weights on [a, b] for current backend."""
        # To remain JAX-jittable we always compute points on the host using NumPy,
        # then convert to the active backend array type exactly once.
        import numpy as _np  # type: ignore

        nodes, weights = _np.polynomial.legendre.leggauss(int(n))
        nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        weights = 0.5 * (b - a) * weights

        return np.asarray(nodes), np.asarray(weights)

    def tablesGL_f(self, precision=[4, 5, 5]):
        precision_key = tuple(int(p) for p in precision)
        cached = self._tables_cache.get(precision_key)
        if cached is not None:
            return cached

        Nphi, Nx, Nmu = precision_key

        phi_roots, phi_weights = self._gauss_legendre(Nphi, 0.0, np.pi)
        x_roots, x_weights = self._gauss_legendre(Nx, -1.0, 1.0)
        mu_roots, mu_weights = self._gauss_legendre(Nmu, -1.0, 1.0)

        phiGL = np.stack((phi_roots, phi_weights), axis=-1)
        xGL = np.stack((x_roots, x_weights), axis=-1)
        muGL = np.stack((mu_roots, mu_weights), axis=-1)

        tables = [phiGL, xGL, muGL]
        self._tables_cache[precision_key] = tables
        return tables

    def tablesGL2_f(self, precision=[10, 10]):   # For Scoccimarro

        precision_key = tuple(int(p) for p in precision)
        cached = self._tables_cache.get(precision_key)
        Nphi,Nmu = precision_key
        Pi= np.pi

        a, b = 0, np.pi
        phi_roots, phi_weights = scipy.special.roots_legendre(Nphi)
        phi_roots = 0.5 * (b - a) * phi_roots + 0.5 * (a + b)
        phi_weights = 0.5 * (b - a) * phi_weights
        phiGL=np.array([phi_roots,phi_weights]).T

        mu_roots, mu_weights = scipy.special.roots_legendre(Nmu)
        muGL=np.array([mu_roots,mu_weights]).T
        tablesGL = [phiGL,muGL]
        self._tables_cache[precision_key] = tablesGL

        return tablesGL


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
        x31AP = -(k1AP + k2AP*x12AP)/k3AP
        x23AP = -(k2AP + k1AP*x12AP)/k3AP

        return (k1AP, k2AP, k3AP,x12AP, x23AP, x31AP,mu1AP, mu2AP, mu3AP,cosphi)

    # def Z2(self, ki, kj, xij, mui, muj, f0, fi, fj, b1, b2, bs,calA=1,calAp=0):

    #     term1 = b2/2 + bs/2 * (xij**2 - 1/3)
    #     km=ki*mui + kj*muj
    #     fij = (fi+fj)/2.0   #This should be f(|ki+kj|)
    #     term2 = km/2 * fij * (mui/ki *  (b1 + fj * muj**2) +
    #                           muj/kj *  (b1 + fi * mui**2) )
    #     F2 = 5/7 + xij/2 * (ki/kj + kj/ki) + 2/7 * xij**2
    #     # G2e = 3/7 + xij/2 * (fi/f0*ki/kj + fj/f0*kj/ki) + 4/7 * xij**2
    #     G2 = (3*(fi+fj)*calA + 3*calAp)/(14*f0) + xij/2 * (fi/f0*ki/kj + fj/f0*kj/ki) + xij**2 * ( (fi+fj)*calA/(2*f0) - (3*(fi+fj)*calA + 3*calAp)/(14*f0) )
    #     term3 = b1 * F2
    #     mu2 = km**2 / (ki**2 + kj**2 + 2 * ki * kj * xij)
    #     term4 = fij * mu2 * G2

    #     return term1 + term2 + term3 + term4

    def Z2(self, ki, kj, xij, mui, muj, f0, fi, fj, b1, b2, bs, calA=1., calAp=0.,
           bfnld=0., alphai=0., alphaj=0.):
        r"""Second-order kernel.  *bfnld* is :math:`b_{\phi\delta} f_\mathrm{NL}`, the
        second-order local-PNG bias, whose kernel is :math:`(\alpha_i + \alpha_j) / 2`."""

        term1 = b2 / 2. + bs / 2. * (xij**2 - 1. / 3.) + bfnld * (alphai + alphaj) / 2.
        km = ki * mui + kj * muj
        fij = (fi + fj) / 2.0  # This should be f(|ki+kj|)
        term2 = km / 2. * fij * (mui / ki * (b1 + fj * muj**2) + muj / kj * (b1 + fi * mui**2))
        F2 = 1. / 2. + 3. * calA / 14. + xij / 2. * (ki / kj + kj / ki) + (1. / 2. - 3. * calA / 14.) * xij**2
        G2 = (3. * calA * (fi + fj) + 3. * calAp) / (14. * f0) + xij / 2. * (fi / f0 * ki / kj + fj / f0 * kj / ki) + ((fi + fj) / (2. * f0) - (3. * calA * (fi + fj) + 3. * calAp) / (14. * f0)) * xij**2
        term3 = b1 * F2
        mu2 = km**2 / (ki**2 + kj**2 + 2. * ki * kj * xij)
        term4 = fij * mu2 * G2
    
        return term1 + term2 + term3 + term4





    def bispectrum(self, k1, k2, x12, mu1, phi, f, sigma2v, Sigma2, deltaSigma2,
                   bpars, qpar, qperp, k_pkl_pklnw_fk, damping = 'lor',interpolation_method= 'cubic',
                   alphak2=None):

        core, (fnl, bphi, bphid), X_FoG_b = split_png_pars(bpars, 7, 3)
        b1, b2, bs, c1, c2, Bshot, Pshot = core

        cosphi = np.cos(phi)
        APtransf = self.APtransforms(k1, k2, x12, mu1, cosphi, qpar, qperp)
        k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP, cosphi = APtransf

        k_     = k_pkl_pklnw_fk[0]
        pkl_   = k_pkl_pklnw_fk[1]
        pklnw_ = k_pkl_pklnw_fk[2]
        fk_    = k_pkl_pklnw_fk[3]

        # alphak2, when given, is alpha(k) k^2 on the k_pkl_pklnw_fk grid: the local-PNG
        # transfer, and what switches the PNG terms on.  See
        # :meth:`NonLinearPowerSpectrumCalculator._initialize_png` and :func:`table_alphak2`.
        png = alphak2 is not None
        # \mathcal{A} factors from eq 4.6 in 2312.10510 (fkpt paper)
        if len(k_pkl_pklnw_fk) >= 6:
            calAarr = k_pkl_pklnw_fk[4]
            calAparr = k_pkl_pklnw_fk[5]
            # Handle both scalar and array-like inputs in a backend-agnostic way (NumPy/JAX).
            if np.ndim(calAarr) > 0 and np.ndim(calAparr) > 0:
                calA = calAarr[0]
                calAp = calAparr[0]
            else:
                calA = calAarr
                calAp = calAparr
        else:
            calA = 1
            calAp=0

        interp_method = interpolation_method
        pk1   = self.interpolation_b(k1AP, k_, pkl_,   method=interp_method)
        pk1nw = self.interpolation_b(k1AP, k_, pklnw_, method=interp_method)
        pk2   = self.interpolation_b(k2AP, k_, pkl_,   method=interp_method)
        pk2nw = self.interpolation_b(k2AP, k_, pklnw_, method=interp_method)
        pk3   = self.interpolation_b(k3AP, k_, pkl_,   method=interp_method)
        pk3nw = self.interpolation_b(k3AP, k_, pklnw_, method=interp_method)

        f1 = self.interpolation_b(k1AP, k_, fk_, method=interp_method)
        f2 = self.interpolation_b(k2AP, k_, fk_, method=interp_method)
        f3 = self.interpolation_b(k3AP, k_, fk_, method=interp_method)

        e1IR = (1 + f1*mu1AP**2 *(2 + f1))*Sigma2 + (f1*mu1AP)**2 * (mu1AP**2 - 1)* deltaSigma2
        e2IR = (1 + f2*mu2AP**2 *(2 + f2))*Sigma2 + (f2*mu2AP)**2 * (mu2AP**2 - 1)* deltaSigma2
        e3IR = (1 + f3*mu3AP**2 *(2 + f3))*Sigma2 + (f3*mu3AP)**2 * (mu3AP**2 - 1)* deltaSigma2

        pkIR1= pk1nw + (pk1-pk1nw)*np.exp(-e1IR*k1AP**2)
        pkIR2= pk2nw + (pk2-pk2nw)*np.exp(-e2IR*k2AP**2)
        pkIR3= pk3nw + (pk3-pk3nw)*np.exp(-e3IR*k3AP**2)


        if png:
            alpha1 = self.interpolation_b(k1AP, k_, alphak2, method=interp_method) / k1AP**2
            alpha2 = self.interpolation_b(k2AP, k_, alphak2, method=interp_method) / k2AP**2
            alpha3 = self.interpolation_b(k3AP, k_, alphak2, method=interp_method) / k3AP**2
        else:
            alpha1 = alpha2 = alpha3 = 0.
        # Local-PNG scale-dependent bias, one per leg.
        bfnld = fnl * bphid
        Z1_1 = b1 + fnl * bphi * alpha1 + f1 * mu1AP**2
        Z1_2 = b1 + fnl * bphi * alpha2 + f2 * mu2AP**2
        Z1_3 = b1 + fnl * bphi * alpha3 + f3 * mu3AP**2

        Z1eft1 = Z1_1 - (c1 * mu1AP**2 + c2 * mu1AP**4) * k1AP**2
        Z1eft2 = Z1_2 - (c1 * mu2AP**2 + c2 * mu2AP**4) * k2AP**2
        Z1eft3 = Z1_3 - (c1 * mu3AP**2 + c2 * mu3AP**4) * k3AP**2


        B12 = (2 * self.Z2(k1AP, k2AP, x12AP, mu1AP, mu2AP, f, f1, f2, b1, b2, bs,calA,calAp, bfnld=bfnld, alphai=alpha1, alphaj=alpha2) * Z1eft1*pkIR1 * Z1eft2*pkIR2)
        B23 = (2 * self.Z2(k2AP, k3AP, x23AP, mu2AP, mu3AP, f, f2, f3, b1, b2, bs,calA,calAp, bfnld=bfnld, alphai=alpha2, alphaj=alpha3) * Z1eft2*pkIR2 * Z1eft3*pkIR3)
        B31 = (2 * self.Z2(k3AP, k1AP, x31AP, mu3AP, mu1AP, f, f3, f1, b1, b2, bs,calA,calAp, bfnld=bfnld, alphai=alpha3, alphaj=alpha1) * Z1eft3*pkIR3 * Z1eft1*pkIR1)
        # Primordial bispectrum; see :meth:`BispectrumCalculator.bispectrum`.
        Bprim = 0.
        if png:
            Bprim = 2. * fnl * Z1eft1 * Z1eft2 * Z1eft3 * (
                png_prim_ratio(alpha1, alpha2, alpha3) * pkIR1 * pkIR2
                + png_prim_ratio(alpha2, alpha3, alpha1) * pkIR2 * pkIR3
                + png_prim_ratio(alpha3, alpha1, alpha2) * pkIR3 * pkIR1)

        W = fog_damping((k1AP * mu1AP, X_FoG_b), (k2AP * mu2AP, X_FoG_b), (k3AP * mu3AP, X_FoG_b),
                        f=f, sigma2v=sigma2v, damping=damping)


        if not getattr(self, '_printed_model_damping_bk', False):
            print(f"[FOLPS] Model Bk: {self.model}, Damping: {damping}")
            self._printed_model_damping_bk = True


        # if self.model == "EFT":
        #     W = 1
        # elif self.model == "TNS":
        #     global use_TNS_model_status
        #     if use_TNS_model_status is not True:
        #         raise RuntimeError("[FOLPS] TNS: set use_TNS_model=True in MatrixCalculator.")
        #     if damping == 'lor':
        #         W = Wlor
        #     elif damping == 'vdg':
        #         W = Winfty
        #     elif damping == 'exp':
        #         W = Wexp
        #     else:
        #         W = Wlor
        # elif self.model == "FOLPSD":
        #     if damping not in ['lor', 'vdg','exp']:
        #         print("[FOLPS] For FOLPSD you must specify a valid damping ('lor', 'vdg','exp'). Default: 'lor'.")
        #         damping = 'lor'
        #     if damping == 'lor':
        #         W = Wlor
        #     elif damping == 'vdg':
        #         W = Winfty
        #     elif damping == 'exp':
        #         W = Wexp
        #     else:
        #         W = Wlor
        # else:
        #     if damping == 'lor':
        #         W = Wlor
        #     elif damping == 'vdg':
        #         W = Winfty
        #     elif damping == 'exp':
        #         W = Wexp
        #     else:
        #         W = Wlor

        ## Noise
        # To match eq.3.14 of 2110.10161, one makes (1+Pshot) -> (1+Pshot)/bar-n; Bshot -> Bshot/bar-n
        shot = (
                  (b1*Bshot + 2.0*Pshot*f1*mu1AP**2) * Z1eft1 * pkIR1
                + (b1*Bshot + 2.0*Pshot*f2*mu2AP**2) * Z1eft2 * pkIR2
                + (b1*Bshot + 2.0*Pshot*f3*mu3AP**2) * Z1eft3 * pkIR3
                + Pshot**2
        )

        bispectrum = W*(B12 + B23 + B31 + Bprim) + shot
        alpha = qpar * qperp**2
        bispectrum = bispectrum / alpha**2

        return bispectrum

    def interpolation_b(self, k_out, k_in, pk_in, method='cubic'):
        """Interpolation function

        Parameters:
        -----------
        k_out : array-like
            Output k values where interpolation is desired
        k_in : array-like
            Input k values
        pk_in : array-like
            Input power spectrum values
        method : str
            Interpolation method: 'linear' or 'cubic' (default)

        Returns:
        --------
        pk_out : array-like
            Interpolated power spectrum values
        """
        if method == 'linear':
            pk_out = np.interp(k_out, k_in, pk_in)
        elif method == 'cubic':
            if backend == 'jax':
                # Use interpax cubic2 which matches SciPy CubicSpline exactly
                pk_out = interp(k_out, k_in, pk_in, method='cubic2')
            else:
                # NumPy backend: use SciPy CubicSpline with not-a-knot boundary conditions
                from scipy.interpolate import CubicSpline
                spline = CubicSpline(k_in, pk_in, bc_type='not-a-knot', extrapolate=True)
                pk_out = spline(k_out)
        else:
            # Default to linear if method not recognized
            pk_out = np.interp(k_out, k_in, pk_in)

        return pk_out

    def sigmas(self, kT,pklT):

        k_BAO = 1/104
        kS = 0.4

        sigma2v_  = simpson(pklT, x=kT) / (6 * np.pi**2)
        sigma2v_ *= 1.05  #correction due to k cut

        # pklT_=pklT[kT<=0.4].copy()
        # kT_=kT[kT<=0.4].copy()

        mask = kT <= 0.4
        pklT_= np.where(mask, pklT, 0.0)
        kT_   = np.where(mask, kT, 0.0)

        j0_small = spherical_jn_backend(0, kT_ / k_BAO)
        j2_small = spherical_jn_backend(2, kT_ / k_BAO)
        Sigma2_ = 1/(6 * np.pi**2)*simpson(pklT_*(1 - j0_small + 2*j2_small), x=kT_)
        j2_full = spherical_jn_backend(2, kT / k_BAO)
        deltaSigma2_ = 1/(2 * np.pi**2)*simpson(pklT*j2_full, x=kT)

        return sigma2v_, Sigma2_, deltaSigma2_


    def angdep_integrands(self, x, mu, phi, cosphi, cos2phi):
        Pi = np.pi

        b000 = 1.0 / (8 * Pi)
        b110 = (-3 * np.sqrt(3) * x) / (8 * Pi)
        b220 = 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * x**2)
        b202 = 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * mu**2)

        b022 = (
            5 * np.sqrt(5) * (
                (-1 + 3 * mu**2) * (-1 + 3 * x**2)
                + 12 * mu * np.sqrt(1 - mu**2)
                * x * np.sqrt(1 - x**2) * cosphi
                + 3 * (-1 + mu**2) * (-1 + x**2) * cos2phi
            )
        ) / (32 * Pi)

        b112 = (
            3 * np.sqrt(2.5) * (
                np.sqrt(3) * (-1 + 3 * mu**2) * x
                + 3 * np.sqrt(3) * mu * np.sqrt(1 - mu**2)
                * np.sqrt(1 - x**2) * cosphi
            )
        ) / (8 * Pi)

        b222 = (
            25 * np.sqrt(70) * (
                1 - 3 * mu**2 * x**2
                - 3 * mu * x * np.sqrt(1 - mu**2)
                * np.sqrt(1 - x**2) * cosphi
                - 1.5 * (1 - mu**2) * (1 - x**2) * (1 - cos2phi)
            )
        ) / (112 * Pi)

        return b000, b110, b220, b202, b022, b112, b222

    def _compute_single_integrand(self, multipole, x, mu, phi, cosphi, cos2phi):
        """
        Compute angular dependence integrand for a single multipole.
        This saves computation time by only calculating what's needed.

        Parameters:
        -----------
        multipole : str
            The multipole to compute: 'B000', 'B110', 'B220', 'B202', 'B022', 'B112'
        x, mu, phi : array-like
            Angular coordinates
        cosphi, cos2phi : array-like
            Pre-computed cos(phi) and cos(2*phi)

        Returns:
        --------
        integrand : array
            The angular integrand for the requested multipole
        """
        Pi = np.pi

        if multipole == 'B000':
            return 1.0 / (8 * Pi)

        elif multipole == 'B110':
            return (-3 * np.sqrt(3) * x) / (8 * Pi)

        elif multipole == 'B220':
            return 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * x**2)

        elif multipole == 'B202':
            return 5 * np.sqrt(5) / (16 * Pi) * (-1 + 3 * mu**2)

        elif multipole == 'B022':
            return (
                5 * np.sqrt(5) * (
                    (-1 + 3 * mu**2) * (-1 + 3 * x**2)
                    + 12 * mu * np.sqrt(1 - mu**2)
                    * x * np.sqrt(1 - x**2) * cosphi
                    + 3 * (-1 + mu**2) * (-1 + x**2) * cos2phi
                )
            ) / (32 * Pi)

        elif multipole == 'B112':
            # The cos(phi) coefficient is 3*sqrt(3), not 6: with mu2 = mu*x + sqrt(1-mu^2)
            # sqrt(1-x^2) cos(phi) the bracket is sqrt(3) * (3 mu mu2 - x), which is the
            # only form symmetric under k1 <-> k2, as (1 1 L) with even L must be.
            return (
                3 * np.sqrt(2.5) * (
                    np.sqrt(3) * (-1 + 3 * mu**2) * x
                    + 3 * np.sqrt(3) * mu * np.sqrt(1 - mu**2)
                    * np.sqrt(1 - x**2) * cosphi
                )
            ) / (8 * Pi)

        elif multipole == 'B222':
            # 125/(8 pi) * S_222; equals 25 sqrt(70)/(112 pi) *
            # [1 - 3 mu^2 x^2 - 3 mu x sqrt(1-mu^2) sqrt(1-x^2) cos(phi)
            #    - (3/2) (1-mu^2)(1-x^2) (1 - cos(2 phi))]
            return (
                25 * np.sqrt(70) * (
                    1 - 3 * mu**2 * x**2
                    - 3 * mu * x * np.sqrt(1 - mu**2)
                    * np.sqrt(1 - x**2) * cosphi
                    - 1.5 * (1 - mu**2) * (1 - x**2) * (1 - cos2phi)
                )
            ) / (112 * Pi)

        else:
            raise ValueError(f"Unknown multipole: {multipole}")


    def Sugiyama_Bl1l2L(self, k1, k2, f, sigma2v, Sigma2, deltaSigma2, bpars, qpar, qper,
                        tablesGL, k_pkl_pklnw_fk, damping='lor', renormalize=True,
                        multipoles=['B000', 'B202'], interpolation_method='linear',precision=None,
                        alphak2=None):
        """
        Compute requested Sugiyama bispectrum multipoles.

        Parameters:
        -----------
        k1, k2 : float or array
            Wavenumbers
        f : float
            Growth rate
        sigma2v, Sigma2, deltaSigma2 : float
            IR resummation scales
        bpars : array-like
            Bias parameters [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b]
        qpar, qper : float
            AP parameters
        tablesGL : list
            Gauss-Legendre tables
        k_pkl_pklnw : list
            [k_array, pkl_array, pklnw_array, fk_array, calA, calAp]
        alphak2 : array, default=None
            alpha(k) k^2 on that same grid, switching the local-PNG terms on;
            see :func:`table_alphak2`.
        damping : str
            Damping type: 'lor', 'exp', 'vdg'
        renormalize : bool
            If True, apply Hl1l2L normalization factors
        multipoles : list of str
            List of multipoles to compute.
            Available: ['B000', 'B110', 'B220', 'B202', 'B022', 'B112']
        interpolation_method : str
            Interpolation method for power spectrum

        Returns:
        --------
        dict : Dictionary with requested multipoles as keys
        """
        Pi = np.pi

        phiGL, xGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        x, wx     = xGL[:, 0],  xGL[:, 1]
        mu, wmu   = muGL[:, 0], muGL[:, 1]

        x_mesh   = x[None, :, None, None]
        mu_mesh  = mu[None, None, :, None]
        phi_mesh = phi[None, None, None, :]

        cosphi  = np.cos(phi_mesh)
        cos2phi = np.cos(2 * phi_mesh)

        bisp = self.bispectrum(
            k1, k2, x_mesh, mu_mesh, phi_mesh,
            f, sigma2v, Sigma2, deltaSigma2,
            bpars, qpar, qper,
            k_pkl_pklnw_fk,
            damping=damping,
            interpolation_method=interpolation_method,
            alphak2=alphak2,
        )

        # Normalization factors for each multipole
        Hl1l2L_dict = {
            'B000': 1.0,
            'B110': -1 / np.sqrt(3),
            'B220': 1 / np.sqrt(5),
            'B202': 1 / np.sqrt(5),
            'B022': 1 / np.sqrt(5),
            'B112': np.sqrt(2 / 15),
            'B222': -2 / np.sqrt(70)
        }

        # Compute only the requested multipoles (saves computation time)
        result = {}
        for mp in multipoles:
            # Calculate integrand only for requested multipole
            integrand = self._compute_single_integrand(mp, x_mesh, mu_mesh, phi_mesh, cosphi, cos2phi)

            # Perform angular integrations
            int_phi = 2 * np.sum(bisp * integrand * wphi[None,None,None,:], axis=3)
            int_mu  = np.sum(int_phi * wmu[None,None,:], axis=2)
            int_x   = np.sum(int_mu * wx[None,:], axis=1)

            if renormalize:
                int_x *= Hl1l2L_dict[mp]

            # Return scalar if k1, k2 are scalars
            if np.ndim(k1) == 0 and np.ndim(k2) == 0:
                result[mp] = float(int_x)
            else:
                result[mp] = int_x

        return result


    def Sugiyama_Bell(self, f, bpars, k_pkl_pklnw_fk,
                      k1k2pairs, qpar, qper, precision=[8, 10, 10], damping='lor',
                      m_bin=None, k_thy=None, do_binning=False, multipoles=['B000', 'B202'],
                      renormalize=True, interpolation_method='linear', bias_scheme='folps',do_interp_bk=False,kout=None,
                      alphak2=None):
        """
        Compute Sugiyama bispectrum multipoles for multiple k1k2 pairs.

        Parameters:
        -----------
        f : float
            Growth rate
        bpars : array-like
            Bias parameters [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b]
        k_pkl_pklnw : list
            [k_array, pkl_array, pklnw_array, fk_array, calA, calAp]
        alphak2 : array, default=None
            alpha(k) k^2 on that same grid, switching the local-PNG terms on;
            see :func:`table_alphak2`.
        k1k2pairs : array-like
            Array of shape (N, 2) with k1, k2 pairs
        qpar, qper : float
            AP parameters
        precision : list
            Gauss-Legendre precision [Nphi, Nx, Nmu]
        damping : str
            Damping type: 'lor', 'exp', 'vdg'
        m_bin : array-like, optional
            Binning matrix (n_bins, n_theory)
        k_thy : array-like, optional
            Theory k values for binning
        do_binning : bool
            If True, perform internal binning
        multipoles : list of str
            Multipoles to compute. Default: ['B000', 'B202']
        renormalize : bool
            If True, apply normalization factors
        interpolation_method : str
            Interpolation method for power spectrum
        bias_scheme : str
            Bias scheme. Default: 'folps' (pat mcdonald bias)

        Returns:
        --------
        tuple : Tuple of arrays, one for each requested multipole
                If do_binning=True, arrays have shape (n_bins,)
                Otherwise, arrays have shape (N,) where N = len(k1k2pairs)
        """
        # Validate requested multipoles
        all_multipoles = ['B000', 'B110', 'B220', 'B202', 'B022', 'B112', 'B222']
        for mp in multipoles:
            if mp not in all_multipoles:
                raise ValueError(f"Invalid multipole '{mp}'. Available: {all_multipoles}")

        bpars = self.set_bias_scheme(bpars, bias_scheme=bias_scheme)
        Pi = np.pi

        # Convert k1k2pairs to array and extract k1, k2 with proper broadcasting dimensions
        k1k2pairs = np.asarray(k1k2pairs)
        k1 = k1k2pairs[:, 0][:, None, None, None]
        k2 = k1k2pairs[:, 1][:, None, None, None]

        # Tables for GL quadrature
        tablesGL = self.tablesGL_f(precision)

        # IR resummation scales
        sigma2v, Sigma2, deltaSigma2 = self.sigmas(k_pkl_pklnw_fk[0], k_pkl_pklnw_fk[1])

        phiGL, xGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        x, wx     = xGL[:, 0],  xGL[:, 1]
        mu, wmu   = muGL[:, 0], muGL[:, 1]

        x_mesh   = x[None, :, None, None]
        mu_mesh  = mu[None, None, :, None]
        phi_mesh = phi[None, None, None, :]

        cosphi  = np.cos(phi_mesh)
        cos2phi = np.cos(2 * phi_mesh)

        # Compute bispectrum for all k1k2 pairs at once (vectorized)
        bisp = self.bispectrum(
            k1, k2, x_mesh, mu_mesh, phi_mesh,
            f, sigma2v, Sigma2, deltaSigma2,
            bpars, qpar, qper,
            k_pkl_pklnw_fk,
            damping=damping,
            interpolation_method=interpolation_method,
            alphak2=alphak2,
        )

        # Normalization factors for each multipole
        Hl1l2L_dict = {
            'B000': 1.0,
            'B110': -1 / np.sqrt(3),
            'B220': 1 / np.sqrt(5),
            'B202': 1 / np.sqrt(5),
            'B022': 1 / np.sqrt(5),
            'B112': np.sqrt(2 / 15),
            'B222': -2 / np.sqrt(70)
        }

        # Compute only the requested multipoles (saves computation time)
        multipoles_dict = {}
        for mp in multipoles:
            # Calculate integrand only for requested multipole
            integrand = self._compute_single_integrand(mp, x_mesh, mu_mesh, phi_mesh, cosphi, cos2phi)

            # Perform angular integrations
            int_phi = 2 * np.sum(bisp * integrand * wphi[None,None,None,:], axis=3)
            int_mu  = np.sum(int_phi * wmu[None,None,:], axis=2)
            int_x   = np.sum(int_mu * wx[None,:], axis=1)

            if renormalize:
                int_x *= Hl1l2L_dict[mp]

            multipoles_dict[mp] = int_x

        # Optional internal binning
        if do_binning:
            if m_bin is None or k_thy is None:
                raise ValueError("do_binning=True requires m_bin and k_thy to be provided.")
            m_bin_arr = np.asarray(m_bin)
            k_thy_arr = np.asarray(k_thy)
            x = np.asarray(k1k2pairs)[:, 0]

            binned_results = []
            for mp in multipoles:
                mp_interp = interp(k_thy_arr, x, multipoles_dict[mp])
                mp_binned = m_bin_arr @ mp_interp
                binned_results.append(mp_binned)

            return tuple(binned_results)

        if do_interp_bk:
            if kout is None:
                raise ValueError("do_interp=True requires kout to be provided.")
            k_thy_arr = np.asarray(kout)
            x = np.asarray(k1k2pairs)[:, 0]
            interp_results = []
            for mp in multipoles:
                mp_interp = interp(k_thy_arr, x, multipoles_dict[mp])
                interp_results.append(mp_interp)

            return tuple(interp_results)

        # Return only requested multipoles
        return tuple(multipoles_dict[mp] for mp in multipoles)


    def Scoccimarro_B024(
        self,
        k1k2k3triplets,
        f,
        sigma2v,
        Sigma2,
        deltaSigma2,
        bpars,
        qpar,
        qperp,
        tablesGL,
        k_pkl_pklnw_fk,
        damping = 'lor',
        interpolation_method='linear',
        multipoles=['B0', 'B2', 'B4'],
        alphak2=None
    ):
        """
        Computation of Scoccimarro B0, B2, B4
        Only computes the multipoles specified in the 'multipoles' parameter.
        """

        twopi = 2.0 * np.pi
        normB0, normB2, normB4 = 0.5, 2.5, 4.5


        k1k2k3triplets = np.asarray(k1k2k3triplets)
        k1 = k1k2k3triplets[:, 0][:, None, None]
        k2 = k1k2k3triplets[:, 1][:, None, None]
        k3 = k1k2k3triplets[:, 2][:, None, None]


        x = (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)

        # --- Gauss Legendre quads ---
        phiGL, muGL = tablesGL
        phi, wphi = phiGL[:, 0], phiGL[:, 1]
        mu,  wmu  = muGL[:, 0],  muGL[:, 1]

        mu_mesh  = mu[None, :, None]     # (1, Nμ, 1)
        phi_mesh = phi[None, None, :]    # (1, 1, Nφ)


        bisp = self.bispectrum(
            k1, k2,
            x,
            mu_mesh,
            phi_mesh,
            f,
            sigma2v,
            Sigma2,
            deltaSigma2,
            bpars,
            qpar,
            qperp,
            k_pkl_pklnw_fk,
            damping = 'lor',
            interpolation_method = interpolation_method,
            alphak2=alphak2,
        )   # (N, Nμ, Nφ)

        # --- φ integration ---
        int_phi = 2.0 * np.sum(bisp * wphi[None, None, :], axis=2)  # (N, Nμ)

        # --- μ integration - Only compute requested multipoles ---
        results = {}

        if 'B0' in multipoles:
            B0 = np.sum(int_phi * wmu[None, :], axis=1) / twopi
            B0 *= normB0
            results['B0'] = B0
        else:
            results['B0'] = None

        if 'B2' in multipoles:
            leg2 = 0.5 * (-1.0 + 3.0 * mu**2)
            B2 = np.sum(int_phi * leg2[None, :] * wmu[None, :], axis=1) / twopi
            B2 *= normB2
            results['B2'] = B2
        else:
            results['B2'] = None

        if 'B4' in multipoles:
            leg4 = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
            B4 = np.sum(int_phi * leg4[None, :] * wmu[None, :], axis=1) / twopi
            B4 *= normB4
            results['B4'] = B4
        else:
            results['B4'] = None

        return results['B0'], results['B2'], results['B4'], x[:, 0, 0]



    def Scoccimarro_Bell(self, k1k2k3triplets, f, bpars, qpar, qperp,
                         k_pkl_pklnw_fk, precision=[10, 10], damping='lor',
                         interpolation_method='linear', m_bin=None, k_thy=None,
                         do_binning=False, multipoles=['B0', 'B2', 'B4'], bias_scheme='folps',
                         alphak2=None):
        """
        Compute Scoccimarro bispectrum multipoles for multiple k1k2k3 triplets.

        Note: Normalization factors (0.5, 2.5, 4.5) are always applied to B0, B2, B4.

        Parameters:
        -----------
        k1k2k3triplets : array-like
            Array of shape (N, 3) with k1, k2, k3 triplets
        f : float
            Growth rate
        bpars : array-like
            Bias parameters [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b]
        qpar, qperp : float
            AP parameters
        k_pkl_pklnw_fk : list
            [k_array, pkl_array, pklnw_array, fk_array]
        precision : list
            Gauss-Legendre precision [Nphi, Nmu]
        damping : str
            Damping type: 'lor', 'exp', 'vdg'
        interpolation_method : str
            Interpolation method for power spectrum
        m_bin : array-like, optional
            Binning matrix (n_bins, n_theory)
        k_thy : array-like, optional
            Theory k values for binning
        do_binning : bool
            If True, perform internal binning
        multipoles : list of str
            Multipoles to compute. Default: ['B0', 'B2', 'B4']
        bias_scheme : str
            Bias scheme. Default: 'folps' (pat mcdonald bias)

        Returns:
        --------
        tuple : (B0, B2, B4, x) if not binning
                (B0_binned, B2_binned, B4_binned) if binning
        """
        bpars = self.set_bias_scheme(bpars, bias_scheme=bias_scheme)

        # IR resummation scales
        kT, pklT = k_pkl_pklnw_fk[0], k_pkl_pklnw_fk[1]
        sigma2v, Sigma2, deltaSigma2 = self.sigmas(kT, pklT)

        # GL tables for (phi, mu)
        tablesGL = self.tablesGL2_f(precision)

        # Fast path: if running with JAX, try to use jax.vmap to compute all
        # triplets at once (avoids Python loops). If that fails, fall back to a
        # list comprehension.
        if backend == 'jax':
            try:
                import jax

                # Build vmapped function that maps over three inputs (k1, k2, k3).
                def _single(k1, k2, k3):
                    # Create a single triplet array
                    triplet = np.array([[k1, k2, k3]])
                    return self.Scoccimarro_B024(
                        triplet, f, sigma2v, Sigma2, deltaSigma2, bpars, qpar, qperp,
                        tablesGL, k_pkl_pklnw_fk, damping=damping,
                        interpolation_method=interpolation_method,
                        multipoles=multipoles, alphak2=alphak2
                    )

                vm = jax.vmap(_single)
                # k1k2k3triplets may be a Python list; convert to backend arrays first
                triplets_arr = np.asarray(k1k2k3triplets)
                k1s = triplets_arr[:, 0]
                k2s = triplets_arr[:, 1]
                k3s = triplets_arr[:, 2]
                stacked = vm(k1s, k2s, k3s)

                # stacked returns tuple (B0, B2, B4, x) for each triplet
                # Each element has shape (N,) with one extra dim from _single
                B0 = stacked[0][:, 0]  # Remove extra dim
                B2 = stacked[1][:, 0]
                B4 = stacked[2][:, 0]
                x = stacked[3][:, 0]

                # Optional internal binning
                if do_binning:
                    if m_bin is None or k_thy is None:
                        raise ValueError("do_binning=True requires m_bin and k_thy to be provided.")
                    m_bin_arr = np.asarray(m_bin)
                    k_thy_arr = np.asarray(k_thy)

                    # Use k1 values for binning
                    k_vals = triplets_arr[:, 0]

                    # Bin each multipole
                    B0_interp = interp(k_thy_arr, k_vals, B0)
                    B0_binned = m_bin_arr @ B0_interp

                    B2_interp = interp(k_thy_arr, k_vals, B2)
                    B2_binned = m_bin_arr @ B2_interp

                    B4_interp = interp(k_thy_arr, k_vals, B4)
                    B4_binned = m_bin_arr @ B4_interp

                    return B0_binned, B2_binned, B4_binned

                return B0, B2, B4, x

            except Exception:
                # If JAX vmapping isn't possible, fall back to list comprehension.
                pass

        # Fallback: call Scoccimarro_B024 with all triplets at once
        B0, B2, B4, x = self.Scoccimarro_B024(
            k1k2k3triplets, f, sigma2v, Sigma2, deltaSigma2, bpars, qpar, qperp,
            tablesGL, k_pkl_pklnw_fk, damping, interpolation_method,
            multipoles=multipoles, alphak2=alphak2
        )

        # Optional internal binning
        if do_binning:
            if m_bin is None or k_thy is None:
                raise ValueError("do_binning=True requires m_bin and k_thy to be provided.")
            m_bin_arr = np.asarray(m_bin)
            k_thy_arr = np.asarray(k_thy)

            # Use k1 values for binning
            triplets_arr = np.asarray(k1k2k3triplets)
            k_vals = triplets_arr[:, 0]

            # Bin each multipole
            B0_interp = interp(k_thy_arr, k_vals, B0)
            B0_binned = m_bin_arr @ B0_interp

            B2_interp = interp(k_thy_arr, k_vals, B2)
            B2_binned = m_bin_arr @ B2_interp

            B4_interp = interp(k_thy_arr, k_vals, B4)
            B4_binned = m_bin_arr @ B4_interp

            return B0_binned, B2_binned, B4_binned

        return B0, B2, B4, x





########################################################
#### Functions for convolving with the window function
########################################################


from scipy.interpolate import RectBivariateSpline


class WindowConvolvedBispectrum:
    """
    Class handling window-function convolution of bispectrum multipoles.
    """

    def __init__(self, model="FOLPSD"):
        self.model = model

    @staticmethod
    def _set_values(arr, rows, cols, values):
        """Set values in a 2D array for both NumPy and JAX arrays."""
        if hasattr(arr, "at"):
            return arr.at[rows, cols].set(values)
        arr[rows, cols] = values
        return arr

    # ============================================================
    # Reconstruction helpers
    # ============================================================

    @staticmethod
    def reconstruct_symmetric(Btri, Nk):
        i, j = np.tril_indices(Nk)
        B = np.zeros((Nk, Nk))
        B = WindowConvolvedBispectrum._set_values(B, i, j, Btri)
        B = WindowConvolvedBispectrum._set_values(B, j, i, Btri)
        return B


    @staticmethod
    def reconstruct_B202_B022(B202_tri, B022_tri, Nk):
        i, j = np.tril_indices(Nk)

        B202 = np.zeros((Nk, Nk))
        B022 = np.zeros((Nk, Nk))

        # Fill lower triangle
        B202 = WindowConvolvedBispectrum._set_values(B202, i, j, B202_tri)
        B022 = WindowConvolvedBispectrum._set_values(B022, i, j, B022_tri)

        # Fill upper triangle (swapped!)
        B202 = WindowConvolvedBispectrum._set_values(B202, j, i, B022_tri)
        B022 = WindowConvolvedBispectrum._set_values(B022, j, i, B202_tri)

        return B202, B022

    # ============================================================
    # Core helper: compute 2D bispectrum grids
    # ============================================================

    def _compute_2D_grids(
        self,
        bispectrum,
        k_ev,
        bisp_nuis_params,
        f,
        qpar,
        qperp,
        k_pkl_pklnw,
        precision,
        renormalize,
        damping,
        interpolation_method,
        bias_scheme,
        multipoles = ['B000', 'B110', 'B220', 'B202', 'B022', 'B112'],
        alphak2 = None
    ):
        Nk = len(k_ev)
        i, j = np.tril_indices(Nk)
        k1k2pairs = np.column_stack((k_ev[i], k_ev[j]))

        Bk2D = bispectrum.Sugiyama_Bell(
            f=f,
            bpars=bisp_nuis_params,
            k_pkl_pklnw=k_pkl_pklnw,
            k1k2pairs=k1k2pairs,
            qpar=qpar,
            qper=qperp,
            precision=precision,
            renormalize=renormalize,
            damping=damping,
            interpolation_method=interpolation_method,
            bias_scheme=bias_scheme,
            multipoles = multipoles,
            alphak2=alphak2)

        B000, B110, B220, B202, B022, B112 = Bk2D

        grids = {
            "B000": self.reconstruct_symmetric(B000, Nk),
            "B110": self.reconstruct_symmetric(B110, Nk),
            "B220": self.reconstruct_symmetric(B220, Nk),
            "B112": self.reconstruct_symmetric(B112, Nk),
        }

        B202g, B022g = self.reconstruct_B202_B022(B202, B022, Nk)
        grids["B202"] = B202g
        grids["B022"] = B022g

        return grids

    # ============================================================
    # Convolution: B000 only
    # ============================================================

    def convolve_B000_diag(
        self,
        bisp_nuis_params,
        bisp_cosmo_params,
        qpar,
        qperp,
        k_pkl_pklnw,
        k_window,
        Ssize,
        window_conv_matrix,
        precision_full=[8, 10, 10],
        precision_diag=[12, 15, 15],
        f=0.0,
        renormalize=True,
        interpolation_method_full="linear",
        interpolation_method_diag="cubic",
        alphak2=None,
        use_full_diag=True,
        get_windowed=True,
        damping="lor",
        bias_scheme="folps",
        multipoles = ['B000', 'B110', 'B220', 'B202', 'B022', 'B112']
    ):
        k_ev = np.logspace(
            np.log10(k_window[0]), np.log10(k_window[-1]), num=Ssize
        )

        bispectrum = BispectrumCalculator(model=self.model)

        grids = self._compute_2D_grids(
            bispectrum,
            k_ev,
            bisp_nuis_params,
            f,
            qpar,
            qperp,
            k_pkl_pklnw,
            precision_full,
            renormalize,
            damping,
            interpolation_method_full,
            bias_scheme,
            multipoles,
            alphak2
        )

        if Ssize == len(k_window):
            Bl1l2L = np.array(
                [grids["B000"], grids["B110"], grids["B220"], grids["B202"]]
            )
            combined = np.concatenate([b.ravel() for b in Bl1l2L])
            conv = window_conv_matrix @ combined
            return np.diag(conv.reshape(len(k_ev), len(k_ev)))

        # Interpolation
        interp = {}
        for key in ["B000", "B110", "B220", "B202", "B022"]:
            spline = RectBivariateSpline(k_ev, k_ev, grids[key], kx=3, ky=3)
            interp[key] = spline(k_window, k_window)

        if not get_windowed:
            return interp["B000"].diagonal()

        if use_full_diag:
            k_diag = np.column_stack([k_window, k_window])
            Bkdiag = bispectrum.Sugiyama_Bell(
                f=f,
                bpars=bisp_nuis_params,
                k_pkl_pklnw=k_pkl_pklnw,
                k1k2pairs=k_diag,
                qpar=qpar,
                qper=qperp,
                precision=precision_diag,
                renormalize=renormalize,
                damping=damping,
                interpolation_method=interpolation_method_diag,
                bias_scheme=bias_scheme,
                multipoles=multipoles,
                alphak2=alphak2)

            B000d, B110d, B220d, B202d, B022d, _ = Bkdiag
            np.fill_diagonal(interp["B000"], B000d)
            np.fill_diagonal(interp["B110"], B110d)
            np.fill_diagonal(interp["B220"], B220d)
            np.fill_diagonal(interp["B202"], B202d)
            np.fill_diagonal(interp["B022"], B022d)

        Bl1l2L = np.array(
            [interp["B000"], interp["B110"], interp["B220"], interp["B202"]]
        )
        combined = np.concatenate([b.ravel() for b in Bl1l2L])
        conv = window_conv_matrix @ combined

        return np.diag(conv.reshape(len(k_window), len(k_window)))

    # ============================================================
    # Convolution: B000 + B202
    # ============================================================

    def convolve_B000_B202_diag(
        self,
        bisp_nuis_params,
        bisp_cosmo_params,
        qpar,
        qperp,
        k_pkl_pklnw,
        k_window,
        Ssize,
        window_conv_matrix_000,
        window_conv_matrix_202,
        precision_full=[8, 10, 10],
        precision_diag=[8, 10, 10],
        f=0.0,
        renormalize=True,
        interpolation_method_full="linear",
        interpolation_method_diag="cubic",
        alphak2=None,
        use_full_diag=True,
        get_windowed=True,
        damping="lor",
        bias_scheme="folps",
        multipoles = ['B000', 'B110', 'B220', 'B202', 'B022', 'B112']
    ):
        k_ev = np.logspace(
            np.log10(k_window[0]), np.log10(k_window[-1]), num=Ssize
        )

        bispectrum = BispectrumCalculator(model=self.model)

        grids = self._compute_2D_grids(
            bispectrum,
            k_ev,
            bisp_nuis_params,
            f,
            qpar,
            qperp,
            k_pkl_pklnw,
            precision_full,
            renormalize,
            damping,
            interpolation_method_full,
            bias_scheme,
            alphak2=alphak2
        )

        interp = {}
        for key in ["B000", "B110", "B220", "B202", "B022", "B112"]:
            spline = RectBivariateSpline(k_ev, k_ev, grids[key], kx=3, ky=3)
            interp[key] = spline(k_window, k_window)

        if not get_windowed:
            return interp["B000"].diagonal(), interp["B202"].diagonal()

        if use_full_diag:
            k_diag = np.column_stack([k_window, k_window])
            Bkdiag = bispectrum.Sugiyama_Bell(
                f=f,
                bpars=bisp_nuis_params,
                k_pkl_pklnw=k_pkl_pklnw,
                k1k2pairs=k_diag,
                qpar=qpar,
                qper=qperp,
                precision=precision_diag,
                renormalize=renormalize,
                damping=damping,
                interpolation_method=interpolation_method_diag,
                bias_scheme=bias_scheme,
                multipoles=multipoles,
                alphak2=alphak2)

            B000d, B110d, B220d, B202d, B022d, B112d = Bkdiag
            np.fill_diagonal(interp["B000"], B000d)
            np.fill_diagonal(interp["B110"], B110d)
            np.fill_diagonal(interp["B220"], B220d)
            np.fill_diagonal(interp["B202"], B202d)
            np.fill_diagonal(interp["B022"], B022d)
            np.fill_diagonal(interp["B112"], B112d)

        Bl000 = np.array(
            [interp["B000"], interp["B110"], interp["B220"], interp["B202"]]
        )
        conv000 = window_conv_matrix_000 @ np.concatenate(
            [b.ravel() for b in Bl000]
        )

        Bl202 = np.array(
            [
                interp["B000"],
                interp["B110"],
                interp["B220"],
                interp["B112"],
                interp["B202"],
            ]
        )
        conv202 = window_conv_matrix_202 @ np.concatenate(
            [b.ravel() for b in Bl202]
        )

        Nk = len(k_window)
        return (
            np.diag(conv000.reshape(Nk, Nk)),
            np.diag(conv202.reshape(Nk, Nk)),
        )


    def reduced_Bl1l2L(
        self,
        bisp_nuis_params,
        bisp_cosmo_params,
        qpar,
        qperp,
        k_pkl_pklnw,
        k_window,
        Ssize,
        precision_full=[8, 10, 10],
        precision_diag=[8, 10, 10],
        f=0.0,
        renormalize=True,
        interpolation_method_full="linear",
        interpolation_method_diag="cubic",
        alphak2=None,
        use_full_diag=True,
        get_windowed=True,
        damping="lor",
        bias_scheme='folps',  #Mod Dic 22
    ):
        k_ev = np.logspace(
            np.log10(k_window[0]), np.log10(k_window[-1]), num=Ssize
        )

        bispectrum = BispectrumCalculator(model=self.model)

        grids = self._compute_2D_grids(
            bispectrum,
            k_ev,
            bisp_nuis_params,
            f,
            qpar,
            qperp,
            k_pkl_pklnw,
            precision_full,
            renormalize,
            damping,
            interpolation_method_full,
            bias_scheme,  #Mod Dic 22
            alphak2=alphak2,
        )


        keys =  ["B000", "B110", "B220", "B202", "B022", "B112"]
        # for key in ["B000", "B110", "B220", "B202", "B022", "B112"]:
        #     spline = RectBivariateSpline(k_ev, k_ev, grids[key], kx=3, ky=3)
        #     interp[key] = spline(k_window, k_window)
        interpb = {
            key: interp2d(k_window, k_window, k_ev, k_ev, grids[key])
            for key in keys
        }


        if not get_windowed:
            return interpb["B000"].diagonal(), interpb["B202"].diagonal()

        if use_full_diag:
            k_diag = np.column_stack([k_window, k_window])
            # Bkdiag = bispectrum.Sugiyama_Bl1l2L(
            #     k_diag,
            #     f,
            #     bisp_nuis_params,
            #     qpar,
            #     qperp,
            #     k_pkl_pklnw=k_pkl_pklnw,
            #     precision=precision_diag,
            #     renormalize=renormalize,
            #     damping=damping,
            #     interpolation_method=interpolation_method_diag,
            #     bias_scheme=bias_scheme,  #Mod Dic 22
            # )
            Bkdiag = bispectrum.Sugiyama_Bell(
                f=f,
                bpars=bisp_nuis_params,
                k_pkl_pklnw=k_pkl_pklnw,
                k1k2pairs=k_diag,
                qpar=qpar,
                qper=qperp,
                precision=precision_diag,
                renormalize=renormalize,
                damping=damping,
                interpolation_method=interpolation_method_diag,
                bias_scheme=bias_scheme,
                multipoles= ['B000', 'B110', 'B220', 'B202', 'B022', 'B112'],
                alphak2=alphak2)

            B000d, B110d, B220d, B202d, B022d, B112d = Bkdiag
            # np.fill_diagonal(interpb["B000"], B000d)
            # np.fill_diagonal(interpb["B110"], B110d)
            # np.fill_diagonal(interpb["B220"], B220d)
            # np.fill_diagonal(interpb["B202"], B202d)
            # np.fill_diagonal(interpb["B022"], B022d)
            # np.fill_diagonal(interpb["B112"], B112d)

            diag_idx = np.arange(interpb["B000"].shape[0])
            interpb["B000"] = self._set_values(interpb["B000"], diag_idx, diag_idx, B000d)
            interpb["B110"] = self._set_values(interpb["B110"], diag_idx, diag_idx, B110d)
            interpb["B220"] = self._set_values(interpb["B220"], diag_idx, diag_idx, B220d)
            interpb["B202"] = self._set_values(interpb["B202"], diag_idx, diag_idx, B202d)
            interpb["B022"] = self._set_values(interpb["B022"], diag_idx, diag_idx, B022d)
            interpb["B112"] = self._set_values(interpb["B112"], diag_idx, diag_idx, B112d)


        Bl1l2L_reduced = np.array([interpb["B000"],
                interpb["B110"],
                interpb["B220"],
                interpb["B112"],
                interpb["B202"]]
                                 )


        return Bl1l2L_reduced



def convolve_Bl1l2L(bisp_nuis_params_, bisp_cosmo_params_,
                     qpar_, qperp_, k_pkl_pklnw_, k_window_, Ssize_,
                     wmat_000, wmat_202,
                     kout=None,
                     precision_full=[8,10,10], precision_diag=[12,15,15],
                     f=0.7,  #temporal to define function
                     renormalize=True,
                     interpolation_method_full='linear',
                     interpolation_method_diag='linear',
                     alphak2=None,
                     use_full_diag=True,multipoles_for_convolution=None,window_source='Carol'):

    bc = WindowConvolvedBispectrum(model="FOLPSD")

    Bl1l2L = bc.reduced_Bl1l2L(bisp_nuis_params_, bisp_cosmo_params_,
                         qpar_, qperp_, k_pkl_pklnw_, k_window_, Ssize_,
                         precision_full=precision_full, precision_diag=precision_diag,
                         f=f,
                         renormalize=renormalize,
                         interpolation_method_full=interpolation_method_full,
                         interpolation_method_diag=interpolation_method_diag,
                         alphak2=alphak2,
                         use_full_diag=use_full_diag)


    i, j = np.triu_indices(len(k_window_))
    k_window=k_window_
    coords = np.stack((k_window[i], k_window[j]), axis=1)
    bk_000 = Bl1l2L[0][i,j]
    bk_110 = Bl1l2L[1][i,j]
    bk_220 = Bl1l2L[2][i,j]
    bk_112 = Bl1l2L[3][i,j]
    # bk_202 = Bl1l2L[4][i,j]
    # B022 = Bl1l2L[4].T
    # bk_022 = B022[i,j]
    bk_202 = Bl1l2L[4]  #different shape (N,N)
    # print(bk_202.shape)
    bk_022 = bk_202.T   #different shape (N,N)
    # print(bk_022.shape)
    bk_202 = bk_202.reshape(-1)
    # print(bk_202.shape)
    bk_022 = bk_022.reshape(-1)
    # print(bk_022.shape)

    coords_000,coords_110,coords_220,coords_112=coords,coords,coords,coords

    N = k_window.shape[0]

    coords_202 = np.empty((N, N, 2), dtype=k_window.dtype)
    coords_202[..., 0] = k_window[:, None]
    coords_202[..., 1] = k_window[None, :]
    # coords_202[..., 0] = k_window[:, None]   # first component = k_i
    # coords_202[..., 1] = k_window[None, :]   # second component = k_j
    # print(coords_202.shape)
    coords_202 = coords_202.reshape(-1, 2)
    # print(coords_202.shape)
    coords_022 = coords_202

    if WinConvFormulae is None:
        raise ImportError(
            "triumvirate is required for window convolution functionality. "
            "Install it with `python -m pip install triumvirate`."
        )

    winconv_formula_000 = WinConvFormulae({
        '000': [
            ('000', '000', 1.),
            ('110', '110', 1./3.),
            ('220', '220', 1./5.),
            ('022', '022', 1./5.),
            ('202', '202', 1./5.),
            ('112', '112', 1./6.),
        ],
    })

    winconv_formula_202 = WinConvFormulae({
        '202': [
            ('000', '202', 1.),
            ('202', '000', 1.),
            ('110', '112', 1./3.),
            ('112', '110', 1./3.),
            ('202', '202', 2./7.),
            ('022', '220', 1./5.),
            ('220', '022', 1./5.),
            ('112', '112', 1./6.),
        ],
    })


    results_bkk_proxy = {
        Multipole((0, 0, 0)): {
            'coords': coords_000,
            'stats_mean': bk_000,
        },
        Multipole((2, 0, 2)): {
            'coords': coords_202,
            'stats_mean': bk_202,
        },
        Multipole((0, 2, 2)): {
            'coords': coords_022,
            'stats_mean': bk_022,
        },
        Multipole((1, 1, 0)): {
            'coords': coords_110,
            'stats_mean': bk_110,
        },
        Multipole((1, 1, 2)): {
            'coords': coords_112,
            'stats_mean': bk_112,
        },
        Multipole((2, 2, 0)): {
            'coords': coords_220,
            'stats_mean': bk_220,
        },
    }

    k_proxy = None
    bkk_proxy = {}

    for multipole in results_bkk_proxy:

        coords = results_bkk_proxy[multipole]['coords']
        flat   = results_bkk_proxy[multipole]['stats_mean']

        if multipole._issym():   # triangular case
            shape_type = 'triu'
        else:                    # full matrix case
            shape_type = 'full'
        k_proxy, bkk_proxy[multipole] = reshape_threept_datatab_jax(
            coords,
            flat,
            shape=shape_type      # <-- static
        )







    if window_source=='Jaide':

        proxy=[]

        #Concatenate cubic proxy multipoles and flatten the array
        for multipole in winconv_formula_000.multipoles_Q:
            #print(multipole)
            proxy.append(bkk_proxy[multipole].flatten())

        proxy_test= np.concatenate(proxy)
        #Convolve proxy with window matrix
        conv_proxy_000= np.dot(wmat_000, proxy_test)
        conv_proxy_202= np.dot(wmat_202, proxy_test)


        Nk = k_proxy.shape[0]
        Nk= 50
        conv_proxy_2d_000 = conv_proxy_000.reshape((Nk, Nk))
        conv_proxy_2d_202 = conv_proxy_202.reshape((Nk, Nk))

        # print(conv_proxy_2d_000.shape, conv_proxy_2d_202.shape)
        mask_1d = k_proxy < 0.251
        mask_2d = mask_1d[:, None] & mask_1d[None, :]



        B000th_on_mock = interp2d(kout,kout, k_proxy[:Nk], k_proxy[:Nk],conv_proxy_2d_000).diagonal()
        B202th_on_mock = interp2d(kout,kout, k_proxy[:Nk], k_proxy[:Nk], conv_proxy_2d_202).diagonal()

        # print(B000th_on_mock.shape, B202th_on_mock.shape)

    if window_source=="Carol":



        proxy_000=[]
        proxy_202=[]

        #Concatenate cubic proxy multipoles and flatten the array
        for multipole in multipoles_for_convolution['LRG']['NGC']['bin1']['000']:
            proxy_000.append(bkk_proxy[multipole].flatten())

        for multipole in multipoles_for_convolution['LRG']['NGC']['bin1']['202']:
            proxy_202.append(bkk_proxy[multipole].flatten())

        proxy_test_000= np.concatenate(proxy_000)
        proxy_test_202= np.concatenate(proxy_202)
        #Convolve proxy with window matrix
        conv_proxy_000= np.dot(wmat_000, proxy_test_000)
        conv_proxy_202= np.dot(wmat_202, proxy_test_202)


        Nk = k_proxy.shape[0]
        conv_proxy_2d_000 = conv_proxy_000.reshape((Nk, Nk))
        conv_proxy_2d_202 = conv_proxy_202.reshape((Nk, Nk))




        B000th_on_mock = interp2d(kout,kout, k_proxy[:Nk], k_proxy[:Nk],conv_proxy_2d_000).diagonal()
        B202th_on_mock = interp2d(kout,kout, k_proxy[:Nk], k_proxy[:Nk], conv_proxy_2d_202).diagonal()

        # print(B000th_on_mock.shape, B202th_on_mock.shape)


    results=[]
    results.append(B000th_on_mock)
    results.append(B202th_on_mock)
    # return k1_mock_b000, B000th_on_mock, B202th_on_mock, np.diag(Bl1l2L[0]), np.diag(Bl1l2L[4])
    return tuple(results)


def interpolation_l(k_out, k_in, pk_in, method='cubic'):
    """Interpolation function

    Parameters:
    -----------
    k_out : array-like
        Output k values where interpolation is desired
    k_in : array-like
        Input k values
    pk_in : array-like
        Input power spectrum values
    method : str
        Interpolation method: 'linear' or 'cubic' (default)

    Returns:
    --------
    pk_out : array-like
        Interpolated power spectrum values
    """
    if method == 'linear':
        pk_out = np.interp(k_out, k_in, pk_in)
    elif method == 'cubic':
        if backend == 'jax':
            # Use interpax cubic2 which matches SciPy CubicSpline exactly
            pk_out = interp(k_out, k_in, pk_in, method='cubic2')
        else:
            # NumPy backend: use SciPy CubicSpline with not-a-knot boundary conditions
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(k_in, pk_in, bc_type='not-a-knot', extrapolate=True)
            pk_out = spline(k_out)
    else:
        # Default to linear if method not recognized
        pk_out = np.interp(k_out, k_in, pk_in)

    return pk_out
# %%
