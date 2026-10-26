# FOLPS package
import os

from .folps import *
from .cosmo_class import *
from .tools import *

# Only expose JAX helpers when explicitly requested.
if os.environ.get("FOLPS_BACKEND", "numpy").lower() == "jax":
	try:
		from .tools_jax import *
	except Exception:
		# Keep numpy import path usable even if JAX extras are unavailable.
		pass

# Full one-loop line-of-sight selection (EdS, model="EFT"); see docs/selection_1loop_equations.md
from .selection_operators import (  # noqa: E402
    SelectionBias, DJSCoefficients, RestFrameCoefficients, EFTExtension, PhysicalDerivativeBias,
    folps_bias_to_djs, djs_coefficients, rest_frame_coefficients, tree_bA_to_linear_djs,
    physical_to_polynomial_derivatives,
)
from .selection_matrices import get_selection_matrices, SelectionMatrices  # noqa: E402
from .selection_templates import (  # noqa: E402
    SelectionLoopTable, SelectionLoopTables, build_selection_tables, selection_loop,
    selection_loop_increment,
)
