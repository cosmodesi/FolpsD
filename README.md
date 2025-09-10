# FOLPSpipe
This repository contains the latest version of the FOLPS code.

## To-Do List

- [DONE] Test the code again for `Afull=False, True`

- [DONE] Include `bG2` and `bGamma3` rotation

- Update the `jax_tools.py` file      (done! but got some erros when running folps)



------------------------  
        
- Introduce the MG modifications for f(R)

## Building the Documentation

To build the documentation locally (in English):

1. Install the requirements:
  ```bash
  pip install -r requirements.txt
  pip install sphinx sphinx_rtd_theme
  ```
2. Build the HTML documentation:
  ```bash
  cd docs
  make html
  ```
3. Open `docs/_build/html/index.html` in your browser.

The documentation is automatically built in English using [Read the Docs](https://readthedocs.org/).
