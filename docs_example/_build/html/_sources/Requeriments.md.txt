# Requeriments

## Pre-requisites

The only pre-requisites are Python and the Python package manager pip:

**Python 3.9 or later:** The `PBHBeta` library requires Python 3.9 or later to be installed on your system.

**pip package manager:** The pip package manager is used to install `PBHBeta` and its dependencies. It should be included with your Python installation by default. 

To check if you have Python installed, type `python --version` in the shell, and you should get `Python 3.[whatever]`.
Then, type python `-m pip --version` in the shell, and see if you get a proper version line starting with `pip 20.0.0 [...]` or a higher version. If an older version is shown, please update pip with `python -m pip install pip --upgrade`. 
If either Python 3 is not installed, or the pip version check produces a `no module named pip` error, use your system’s package manager or contact your local IT service.

## Installing dependencies in python

Imperative libraries are:

1. **NumPy:** NumPy is a Python library that provides support for working with multidimensional arrays and high-performance mathematical functions. It is widely used in data science and scientific computing due to its efficiency, linear algebra capabilities, and integration with other Python libraries. It is an essential tool for complex mathematical operations and numerical data manipulation in Python.

To install copy-paste this:

```{code-block}
---
emphasize-lines: 1
---
pip install numpy
```

2. **SciPy:** SciPy is a Python library that extends the capabilities of NumPy and provides additional functionalities for scientific computing. It includes tools for integration, optimization, interpolation, signal and image processing, statistical functions, and more.

To install copy-paste this:

```{code-block}
---
emphasize-lines: 1
---
pip install scipy
```

3. **Matplotlib:** This library provides support for creating various types of charts and graphs in Python.

To install copy-paste this:

```{code-block}
---
emphasize-lines: 1
---
pip install matplotlib
```

If you prefer, you can install all in one copy-paste line:

:::{tip}
:name: a-tip-reference
```{code-block} python
:emphasize-lines: 1
pip install matplotlib numpy scipy
```
:::

## Using Google Colab

For easy and immediate use, we recommend using Google Colab. Within it, it is necessary to install some libraries and `PBHBeta` in the first cells.

```{code-block}
%pip install numpy matplotlib scipy
!pip install -e git+https://github.com/TadeoDGAguilar/PBHBeta#egg=PBHBeta
%cd /content/src/pbhbeta
```
