# Requeriments

## Pre-requisites

The only pre-requisites are Python and the Python package manager pip:

**Python 3.9 or later:** The `PBHBeta` library requires Python version 3.9 or later to be installed on your system.

**pip package manager:** The pip package manager is used to install `PBHBeta` and its dependencies. It should be included with your Python installation by default. 

To confirm the presence of Python on your system, enter `python --version` in the command shell; this should yield a response like `Python 3.[whatever]`.

Subsequently, input `python -m pip --version` in the shell to ascertain the version of pip currently installed. Look for a valid version line that starts with `pip 20.0.0 [...]` or a more recent number. If an older version is displayed, execute `python -m pip install pip --upgrade` to update pip.

If either Python 3 is not installed or you encounter a `no module named pip` error during the pip version check, you should consider utilizing your system's package manager or seeking assistance from your local IT service.

## Installing dependencies in python

The required imperative libraries are as follows:

1. **NumPy:** NumPy is a fundamental Python library that facilitates working with multidimensional arrays and high-performance mathematical operations. It is extensively used in data science and scientific computing due to its efficiency, linear algebra capabilities, and seamless integration with other Python libraries. NumPy is essential for intricate mathematical computations and numerical data manipulation.

To install, copy and paste the following command:

```{code-block}
---
emphasize-lines: 1
---
pip install numpy
```

2. **SciPy:** SciPy extends the capabilities of NumPy and offers additional functionalities for scientific computing. It encompasses tools for integration, optimization, interpolation, signal and image processing, statistical functions, and more.

To install, copy and paste the following command:

```{code-block}
---
emphasize-lines: 1
---
pip install scipy
```

3. **Matplotlib:** Matplotlib is a library that enables the creation of diverse types of charts and graphs in Python.

To install, copy and paste the following command:

```{code-block}
---
emphasize-lines: 1
---
pip install matplotlib
```

Alternatively, you can install all these libraries in one command:

:::{tip}
:name: a-tip-reference
```{code-block} python
:emphasize-lines: 1
pip install matplotlib numpy scipy
```
:::

## Using Google Colab

For effortless and immediate usage, we recommend utilizing Google Colab. Within it, you need to install certain libraries and the `PBHBeta` package in the initial cells.

```{code-block}
%pip install numpy matplotlib scipy
!pip install -e git+https://github.com/TadeoDGAguilar/PBHBeta#egg=PBHBeta
%cd /content/src/pbhbeta
```
