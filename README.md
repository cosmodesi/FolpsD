# FOLPSpipe
This repository contains the latest version of the FOLPS code.

## To-Do List

- Include `bG2` and `bGamma3` rotation 

- Update the `jax_tools.py` file      (done! but got some erros when running folps)

- Implement a function to compute the non-wiggle component, including extrapolation:  
  `pnw(k, pk, h)`

- Develop a function to perform IR resummation for the linear power spectrum:  
  `ir_resum_linear(k, pk, h)`

- Introduce the MG modifications for f(R)

- dividir los inputs para las clases nonlinear y RSDmultipolos en dos formas.
  Que reciban de input parametros que NO dependan de la cosmologia,
  minetras que para las funciones que se ejecutan, alli si vayan los parametros que se varian

- Ver la mejor forma de conectar el BackendManager con las funciones del folps.py, quizas:

algo asi: 

class MatrixCalculator:
    def __init__(self, backend_functions):
        """Inicializa la clase con el backend elegido (NumPy/JAX) y sus funciones asociadas."""
        self.backend = backend_functions  