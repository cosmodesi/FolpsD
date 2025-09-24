# Quickstart

Here's a simple example of using `PBHBeta`. Each aspect will be explained in subsequent sections.

## Importing Modules

```{note}
You can verify the correct installation of the `PBHBeta` library by immediately importing it and its modules.
```

**1. Import the `PBHBeta` library and necessary modules:**

```{code-block}
from PBHBeta import *
import matplotlib.pyplot as plt
import numpy as np
```

## Using Basic Functions

Basic functions assist in creating input data arrays and solving ODEs. See [Basic functions](https://pbhbeta.readthedocs.io/en/latest/Module_1.html#basic-functions) for details.

**2. Generate an array of masses using the `put_M_array`{l=python} function:** The initial function to execute is {py:func}`PBHBeta.functions.put_M_array`, which generates an array of masses from a specified minimum and maximum mass value in grams.

```{code-block}
PBHBeta.functions.put_M_array(constants.M_pl_g, 1e20)
```

Note that we use the `constants.py` module to import `constants.M_pl_g` (the Planck mass in grams) for defining the minimum mass used in this example.

```{note}
The `constraints.py` module stores the return value of {py:func}`PBHBeta.functions.put_M_array`.
```

## Importing data

The `PBHBeta` library has a dedicated module `constraints.py` for storing calculated data from basic and main functions (see [functions.py](https://pbhbeta.readthedocs.io/en/latest/Module_1.html)). 
This facilitates convenient access to data for use as inputs in classes. Detailed examples are in the [Jupyter notebooks](../examples.md) section.

**3. Access generated masses stored in the `constraints.M_tot`:** Retrieve values from `constraint.M_tot` and store them for operations within `PBHBeta` functions.

```{code-block} python
:lineno-start: 1
M_tot = constraints.M_tot
```

## Using Main Functions

The [main functions](https://pbhbeta.readthedocs.io/en/latest/Module_1.html#main-functions) in `functions.py` calculate specific PBH abundance constraints.


**4. Calculate abundances (betas) considering dark matter (DM) constraints using the {py:func}`PBHBeta.functions.Betas_DM()` function:**


```{code-block} python
:lineno-start: 2
M_n, betas, M_relic, betas_relic = functions.Betas_DM(M_tot)
```

```{note}
`PBHBeta.functions.Betas_DM()` stores the total abundance values in an array named `betas_DM_tot` within the `constraints.py` module. Additionally, the function returns several arrays:

`M_n`: Masses of PBHs that are currently present and could be potential candidates for dark matter.
`betas`: Corresponding abundance values for the PBHs mentioned in `M_n`.
`M_relic`: Masses of PBHs that might have left behind stable relics, which could contribute to dark matter after undergoing evolution and evaporation.
`betas_relic`: Associated abundance values for the PBHs referred to in `M_relic`. 

To manipulate these arrays, you can directly access them by invoking `functions.Betas_DM(M_tot)`[number], where [number] is 0, 1, 2, or 3, corresponding to each output 
array, respectively.
```

```{hint}
For more information on parameters and outputs, click the bolded declared functions.
```

## Outputs and plots

**5. Plot the calculated abundances using matplotlib:** Here, we show constraints on monochromatic PBH abundance as a function of mass in the standard Big Bang scenario.


```{code-block} python
:lineno-start: 3
plt.loglog(M_n, betas, "k:",label=r"$\Omega_{DM}$")
plt.fill_between(M_n, betas, betas*0+10, color='grey',alpha=0.2)
plt.loglog(M_relic, betas_relic, "k--",label=r"$\Omega_{DM}$")
plt.fill_between(M_relic, betas_relic, betas_relic*0+10, color='grey',alpha=0.2)

plt.xlabel(r"$M_{\rm PBH}~[\rm{g}]$")
plt.ylabel(r"$\beta$")

plt.legend(ncol=1)
plt.show()
```

`out:`
```{figure} img/IMG_DM.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```

If the displayed graph is obtained, congratulations! You have successfully installed and used `PBHBeta`. Explore more functionalities and examples in the [Jupyter notebooks](../examples.md) section.

