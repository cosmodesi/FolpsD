# Quickstart

Let us present here a simple example, without explaining much about it — each of its aspects will be broken down in the following sections.

## Importing Modules

```{note}
One way to verify that the `PBHBeta` library has been installed correctly is to immediately import the library and all its modules.
```

1. Import the `PBHBeta` library and any required modules:

```{code-block}
from PBHBeta import *
import matplotlib.pyplot as plt
import numpy as np
```

## Call a basic function

The basic functions have the main task of assisting the user in constructing the input data array, as well as solving ODEs
(see [Basic functions](https://pbhbeta.readthedocs.io/en/latest/Module_1.html#basic-functions) section).

2. Generate an array of masses using the `put_M_array`{l=python} function:

The first function to execute is {py:func}`PBHBeta.functions.put_M_array`, which contains the instruction to generate an array of masses from a maximum value of mass in 
grams.

```{code-block}
PBHBeta.functions.put_M_array(1e20)
```
```{note}
The library contains a module called `constraints.py`, where the `return` of {py:func}`PBHBeta.functions.put_M_array` will be stored.
```

## Import data from a module

It is important to say that the PBHBeta library has a dedicated module (`constraints.py`) for storing data that has been calculated 
using both the basic and main functions (see [functions.py](https://pbhbeta.readthedocs.io/en/latest/Module_1.html)). 
This is done to provide greater convenience when accessing and reading such 
data for use as inputs in classes. Some examples more detailed are presented in the [Jupyter notebooks](../examples.md) section.

3. Access the generated masses stored in the `constraints.M_tot` variable:

You can retrieve the values stored in `constraint.M_tot` and store them in a variable to perform operations within the PBHBeta functions. 

Here's an example:

```{code-block} python
:lineno-start: 1
M_tot = constraints.M_tot
```

## Use a main function

The [main functions](https://pbhbeta.readthedocs.io/en/latest/Module_1.html#main-functions) within the `functions.py`
module aim to perform specific calculations on the constraints of PBH abundance.


4. Calculate the abundances (betas) considering the dark matter (DM) constraints using the Betas_DM function {py:func}`PBHBeta.functions.Betas_DM()` with input `M_tot`:

```{code-block} python
:lineno-start: 2
functions.Betas_DM(M_tot)
```

```{note}
The documentation of the `PBHBeta.functions.Betas_DM()` function specifies that it stores the values of the total abundance in an array called `betas_DM_tot` within the 
`constraints.py` module. Additionally, the function returns four arrays, namely `M_n`, `betas`, `M_relic`, and `betas_relic`.

To manipulate these arrays, you can directly access them by invoking `functions.Betas_DM(M_tot)`[number], where `number` can be 0, 1, 2, or 3, corresponding to each output 
array respectively.

```

## Outputs and plot

Lastly, it only remains to retrieve the data and plot them. In this case, we will obtain the constraints on the abundance of monochromatic PBHs as a function of mass in the standard Big Bang scenario.

5. Plot the calculated abundances using matplotlib:

```{code-block} python
:lineno-start: 3
plt.loglog(functions.Betas_DM(M_tot)[0],functions.Betas_DM(M_tot)[1], "k:",label=r"$\Omega_{DM}$")
plt.fill_between(functions.Betas_DM(M_tot)[0],functions.Betas_DM(M_tot)[1], functions.Betas_DM(M_tot)[1]*0+10, color='grey',alpha=0.2)
plt.loglog(functions.Betas_DM(M_tot)[2],functions.Betas_DM(M_tot)[3], "k:",label=r"$\Omega_{DM}$")
plt.fill_between(functions.Betas_DM(M_tot)[2],functions.Betas_DM(M_tot)[3], functions.Betas_DM(M_tot)[3]*0+10, color='grey',alpha=0.2)
plt.xlabel(r"$M_{\rm PBH}~[\rm{g}]$")
plt.ylabel(r"$\beta$")
plt.xlim([1,1e20])
plt.ylim([1e-30,1])

plt.legend()
plt.show()
```

`out:`
```{figure} img/IMG_DM.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```

If you have obtained the displayed graph
 congratulations! You have successfully installed and used `PBHBeta`. You can explore more functionalities and examples into [Jupyter notebooks](../examples.md) section.

