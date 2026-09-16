# FOLPS package
import os

from .folps import *
from .cosmo_class import *
from .tools import *
from .png import alpha_png, primordial_pk, bphi_universality, bfnl_loc

# Only expose JAX helpers when explicitly requested.
if os.environ.get("FOLPS_BACKEND", "numpy").lower() == "jax":
	try:
		from .tools_jax import *
	except Exception:
		# Keep numpy import path usable even if JAX extras are unavailable.
		pass
