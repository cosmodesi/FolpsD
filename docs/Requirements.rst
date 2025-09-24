Requirements
============

Minimum prerequisites
---------------------
- Python 3.8+ recommended.
Core Python packages (see `requirements.txt`) include:

- numpy
Optional / heavier dependencies (needed for full functionality):

- `classy` (CLASS python wrapper) — used to compute linear power spectra.
Installing everything (recommended for full features):

.. code-block:: bash

   python -m pip install -r requirements.txt

Using Google Colab or cloud environments
----------------------------------------

.. code-block:: python

   %pip install numpy scipy matplotlib
   %pip install -e git+https://github.com/<your-username>/FOLPSpipe#egg=FOLPSpipe
