# FOLPSpipe
This repository contains the latest version of the FOLPS code.

## To-Do List

- [DONE] Test the code again for `Afull=False, True`

- [DONE] Include `bG2` and `bGamma3` rotation

- [DONE] Include a flag to turn on/off IR-resummation (see the `FOLPSnu_v3_May25` pipeline)

- Update the `jax_tools.py` file      (done! but got some erros when running folps)

- [DONE] Split the inputs for the `NonLinear` and `RSDMultipoles` classes into two types:
  - Inputs that do **not** depend on cosmology should be passed to the class constructor.
  - Cosmology-dependent parameters should be passed directly to the functions that are executed.

- [DONE] Explore the best way to connect the `BackendManager` with the functions in `folps.py`, perhaps something like:

```python
class MatrixCalculator:
    def __init__(self, backend_functions):
        """Inicializa la clase con el backend elegido (NumPy/JAX) y sus funciones asociadas."""
        self.backend = backend_functions
```
- [DONE] Check Bispectrum and optimize it        
        
- [DONE] Include VDG models (standard and keeping deltaP)      

------------------------  
        
- Introduce the MG modifications for f(R)
