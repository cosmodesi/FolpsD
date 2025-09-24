
Installation
============

There are several ways to install and run FOLPSpipe. Choose the one that fits your workflow.

1. Clone the repository and install requirements (recommended for development):

.. code-block:: bash

   git clone <your-repo-url>
   cd FOLPSpipe
   python -m pip install -r requirements.txt

2. Install in editable mode (developer install):

.. code-block:: bash

   pip install -e .

3. Install from a GitHub URL (if published):

.. code-block:: bash

   pip install git+https://github.com/<your-username>/FOLPSpipe.git

Notes
-----
- For full functionality (CLASS, JAX), install the optional dependencies listed in `requirements.txt`.
- Use a virtual environment to avoid conflicting system packages.
