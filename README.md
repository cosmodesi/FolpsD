# FOLPSpipe
This repository contains the latest version of the FOLPS code.

## To-Do List

- Test the code again (I found a bug recently), test it for `Afull=False, True`

- Include `bG2` and `bGamma3` rotation 

- Update the `jax_tools.py` file      (done! but got some erros when running folps)

- dividir los inputs para las clases nonlinear y RSDmultipolos en dos formas.
  Que reciban de input parametros que NO dependan de la cosmologia,
  minetras que para las funciones que se ejecutan, alli si vayan los parametros que se varian

- Ver la mejor forma de conectar el BackendManager con las funciones del folps.py, quizas:

algo asi: 

```python
class MatrixCalculator:
    def __init__(self, backend_functions):
        """Inicializa la clase con el backend elegido (NumPy/JAX) y sus funciones asociadas."""
        self.backend = backend_functions
```
- Check Bispectrum and optimize it        
        
- Introduce the MG modifications for f(R)
