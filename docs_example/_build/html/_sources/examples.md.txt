# Jupyter notebooks

## Primordial Black Holes in non-standard cosmology

### Early matter dominated scenario


Constraints on the abundance of PBHs as a function of their mass.
`{math}$N_{MD}$ is the total number of e-folds that the MD epoch lasted.

### Scalar field dominated scenario

### Stiff fluid dominated scenario

```{code-block} python
:lineno-start: 1
from PBHBeta import *
import matplotlib.pyplot as plt
import numpy as np
```

```{code-block} python
:lineno-start: 4
functions.put_M_array(1e20)
```

```{code-block} python
:lineno-start: 5
M_tot = np.array(constraints.M_tot)
w = 1/3 # omega from Equation of State

funtions.Betas_DM(M_tot)
funtions.Betas_BBN(M_tot,w)
funtions.Betas_CMB_AN(M_tot,w)
funtions.Betas_GRB(M_tot,w)
funtions.Betas_Reio(M_tot,w)
funtions.Betas_SD(M_tot,w)
funtions.Betas_LSP(M_tot)
```

```{code-block} python
:lineno-start: 15
funtions.get_Betas_full(M_tot)
```

```{code-block} python
:lineno-start: 16
plt.loglog(M_tot,constraints.betas_full,"k:")
plt.ylim([1e-30,1e-13])
plt.xlim([1e-2,1e20])
plt.show()
```
`out:`
```{figure} img/betas_full.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```

## Computing abundances with efolds

$\beta_{PBH} = \beta(N_{\rm reh}, \omega, \gamma_{\rm reh})$

```{code-block} python
:lineno-start: 1
from betaPBH import functions, constants, constraints, BfN, BfS
import matplotlib.pyplot as plt
import numpy as np
```

```{code-block} python
:lineno-start: 4
functions.put_M_array(0.0123)
M_tot = np.array(constraints.M_tot)
```

```{code-block}
:lineno-start: 6
plt.loglog(M_tot,BfN.get_betas_reh_tot(10,0,1),label = r"$N_{\rm reh}=10$")
plt.loglog(M_tot,BfN.get_betas_reh_tot(20,0,1),label = r"$N_{\rm reh}=20$")
plt.loglog(M_tot,BfN.get_betas_reh_tot(30,0,1),label = r"$N_{\rm reh}=30$")
plt.ylim([1e-30,1])
plt.xlim([1,1e20])
plt.xlabel(r"$M_{\rm PBH}~[\rm{g}]$")
plt.ylabel(r"$\beta$")
plt.legend(ncol=2,bbox_to_anchor=(0.85, 1.5))
plt.show()
```

`out:`
```{figure} img/BfN.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```


## Computing abundances with Stiff

$\beta_{PBH} = \beta(N_{\rm stiff}, \omega, \gamma_{\rm stiff})$

```{code-block} python
:lineno-start: 1
from betaPBH import functions, constants, constraints, BfN, BfS
import matplotlib.pyplot as plt
import numpy as np
```

```{code-block} python
:lineno-start: 4
functions.put_M_array(0.0123)
M_tot = np.array(constraints.M_tot)
```

```{code-block}
:lineno-start: 6
plt.loglog(M_tot,BfS.get_betas_stiff_tot(5,1.,1.),"k",label = r"$N_{\rm stiff} = 5$",alpha = 0.8)
plt.loglog(M_tot,BfS.get_betas_stiff_tot(10,1.,1.),"k",label = r"$N_{\rm stiff} = 10$",alpha = 0.8)
plt.loglog(M_tot,BfS.get_betas_stiff_tot(15,1.,1.),"k",label = r"$N_{\rm stiff} = 15$",alpha = 0.8)
plt.xlabel(r"$M_{\rm PBH}~[\rm{g}]$")
plt.ylabel(r"$\beta$")
plt.ylim([1e-40,1e-5])
plt.xlim([1,1e20])
plt.legend(ncol=2,bbox_to_anchor=(0.85, 1.5))
plt.show()
```
`out:`
```{figure} img/BfS.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```

## Computing K from Number of Reheating (KfN)

```{code-block} python
:lineno-start: 1
from PBHBeta import *
import matplotlib.pyplot as plt
import numpy as np
```

```{code-block} python
:lineno-start: 4
delta_c = 0.41
sigma_tot = np.array(functions.inverse_error(betas_full,delta_c))
k_md = KfN.k_MD(M_tot)
```

```{code-block}
:lineno-start: 7
sigma_10k, k_10s = KfN.get_k_Nreh(M_tot,10,0,1)
sigma_20k, k_20s = KfN.get_k_Nreh(M_tot,20,0,1)
sigma_30k, k_30s = KfN.get_k_Nreh(M_tot,30,0,1)
```

```{code-block}
:lineno-start: 10

B = 25/4

plt.loglog(k_md,B*sigma_tot**2,"k",label = "SBB",color='#0000FF')
plt.fill_between(k_md, B*sigma_tot**2, (B*sigma_tot**2)*0+10, color='#0000FF',alpha=0.2)

plt.loglog(k_10s,(B*sigma_10k**2),"k",color='lime' ,label = r"$N_{\rm MD}=10$")
plt.fill_between(k_10s, (B*sigma_10k**2), (B*sigma_10k**2)[:567]*0+10, color='lime',alpha=0.2)

plt.loglog(k_20s,(B*sigma_20k**2),"k", label = r"$N_{\rm MD}=20$", color='red')
plt.fill_between(k_20s, (B*sigma_20k**2), (B*sigma_20k**2)*0+10, color='red',alpha=0.2)

plt.loglog(k_30s,(B*sigma_30k**2),"k", label = r"$N_{\rm MD}=30$", color='brown')
plt.fill_between(k_30s, (B*sigma_30k**2), (B*sigma_30k**2)*0+10, color='brown',alpha=0.2)

plt.loglog(k_md,B*sigma_tot**2, "k", color='black')
plt.xlabel(r"$k~[\rm{Mpc}^{-1}]$")
plt.ylabel(r"$\mathcal{P}_\zeta(k)$")
plt.ylim([1e-7,1])
plt.xlim(0.45e13,5.3e22)
plt.legend(ncol=2,bbox_to_anchor=(0.85, 1.23))
plt.show()
```

`out:`
```{figure} img/KfN.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```

## Computing K from Stiff Fluid-Dominated (KfS)

```{code-block} python
:lineno-start: 1
from PBHBeta import *
import matplotlib.pyplot as plt
import numpy as np
```

```{code-block} python
:lineno-start: 4
delta_c = 0.41
sigma_tot = np.array(functions.inverse_error(betas_full,delta_c))
k_phys_rad = np.array(KfS.k_rad(M_tot))
```

```{code-block} python
:lineno-start: 7
sigma_5st, k_5st = KfS.get_k_SD(M_tot,5,1,1)
sigma_10st, k_10st = KfS.get_k_SD(M_tot,10,1,1)
sigma_15st, k_15st = KfS.get_k_SD(M_tot,15,1,1)
```

```{code-block} python
:lineno-start: 10
C = (8/2)**2/4

plt.loglog(k_phys_rad, A*sigma_tot**2, "k", color='black', label = "SBB")
plt.fill_between(k_phys_rad, A*sigma_tot**2, [10]*len(A*sigma_tot**2), color='black',alpha=0.2)

plt.loglog(k_5st,(C*sigma_5st**2),"r", color='red', label = r"$N_{\rm SD}=5$")
plt.fill_between(k_5st, (C*sigma_5st**2), (C*sigma_5st**2)*0+10, color='red',alpha=0.2)

plt.loglog(k_10st,(C*sigma_10st**2),"g", color='green',label = r"$N_{\rm SD}=10$")
plt.fill_between(k_10st, (C*sigma_10st**2), (C*sigma_10st**2)*0+10, color='green',alpha=0.2)

plt.loglog(k_15st,(C*sigma_15st**2),"b", color='blue',label = r"$N_{\rm SD}=15$")
plt.fill_between(k_15st, (C*sigma_15st**2), (C*sigma_15st**2)*0+10, color='blue',alpha=0.2)

plt.xlabel(r"$k~[\rm{Mpc}^{-1}]$")
plt.ylabel(r"$\mathcal{P}_\zeta(k)$")
plt.ylim([6.5e-3,2e-2])
plt.xlim([4.3e12,1e26])
plt.ylabel(r"$\mathcal{P}_\zeta(k)$")
plt.legend(ncol=2,bbox_to_anchor=(0.85, 1.25))

plt.show()
```
`out:`
```{figure} img/KfS.png
:alt: fishy
:class: bg-primary
:width: 500px
:align: center
```



