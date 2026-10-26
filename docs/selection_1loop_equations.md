# One-loop line-of-sight selection in FolpsD: equations and conventions

Status: implemented on branch `feature/selection-1loop-eds` (base commit `6012d6d`).
This document supersedes the pre-implementation note
`folps_orientation_selection_1loop_equations_audited.md` (2026-10-26) and records the
conventions, equations and derived results *as implemented*. Validation evidence is in
[`selection_1loop_validation.md`](selection_1loop_validation.md); sources, versions and
recorded corrections in [`selection_1loop_sources.md`](selection_1loop_sources.md).

**Scope.** Renormalized one-loop galaxy **auto** power spectrum in redshift space for
Gaussian initial conditions, a single pressureless fluid, the plane-parallel limit,
**EdS kernels with scale-independent growth**, FolpsD `model="EFT"` (no phenomenological
FoG factor), BAO IR resummation and Alcock-Paczynski (AP) projection retained. All nine
independent deterministic line-of-sight (LOS) selection operators through third order
contribute through both P22 and P13, including their interference with ordinary bias and
RSD. The `fk` (massive-neutrino / scale-dependent growth) generalization is **not**
implemented (section 12).

Primary references: Desjacques, Jeong & Schmidt (DJS), arXiv:1806.04015v2; Tomlinson,
Grasshorn Gebhardt & Jeong, arXiv:2004.03629; FolpsD, arXiv:2208.02791 and 2604.08895.

---

## 1. Conventions

\[
F(\mathbf k)=\int d^3x\,e^{-i\mathbf k\cdot\mathbf x}F(\mathbf x),\qquad
\int_{\mathbf q}\equiv\int\frac{d^3q}{(2\pi)^3},\qquad
\langle\delta_L(\mathbf k)\delta_L(\mathbf k')\rangle=(2\pi)^3\delta_D(\mathbf k+\mathbf k')P_L(k).
\]

Conformal time \(\tau\), \(\mathcal H=aH\), peculiar velocity \(v_i\), \(u_i=v_i/\mathcal H\)
(a length). \(\partial_\parallel=n_i\partial_i\), \(\mu=\hat{\mathbf k}\cdot\mathbf n\),
\(\eta=\partial_\parallel u_\parallel\), \(R_{ij}(\mathbf k)=k_ik_j/k^2\),
\(\mathcal D_{ij}=\partial_i\partial_j/\nabla^2-\delta_{ij}/3\), \(K_{ij}=\mathcal D_{ij}\delta\).
The velocity divergence \(\vartheta=-\nabla\cdot\mathbf v/(\mathcal Hf)\) has EdS kernels
\(G_n\), so

\[
u_i^{(n)}(\mathbf K)=if\frac{K_i}{K^2}G_n,\qquad
\eta^{(n)}=-f\mu_K^2G_n,\qquad
s^{(1)}_i(\mathbf K)=i\frac{K_i}{K^2}\delta_L .
\]

Perturbative expansions carry **no external \(1/n!\)**; kernels are symmetrized, and `Sym`
is the *normalized* average over distinct assignments of momenta to the factors of a
product (an explicit factor such as the 2 in \(2\eta\eta^{(2)}\) is not supplied by `Sym`).

Variances (never all called `sigma2`):

\[
\sigma_0^2=\int_{\mathbf q}P_L,\qquad
\sigma_{-1}^2=\int_{\mathbf q}\frac{P_L}{q^2}=\frac{1}{2\pi^2}\int dq\,P_L,\qquad
\sigma_\psi^2=\tfrac13\sigma_{-1}^2 \;(\text{FolpsD's } \texttt{sigma2psi}).
\]

All are evaluated on FolpsD's extrapolated input range \(10^{-5}\le q\le200\,h/\)Mpc.
Units: \(k\) in \(h\)/Mpc, \(P\) in (Mpc/\(h\))\(^3\).

## 2. Operator basis, coefficients and interface

Deterministic operators relevant to the one-loop auto spectrum:

\[
\mathfrak O_1=\{\delta,\eta\},\quad
\mathfrak O_2=\{\delta^2,K^2,\delta\eta,\eta^2,(KK)_\parallel,\Pi^{[2]}_\parallel\},\quad
\mathfrak O_3=\{O_{\rm td},\delta\Pi^{[2]}_\parallel,\eta\Pi^{[2]}_\parallel,(K\Pi^{[2]})_\parallel,\Pi^{[3]}_\parallel\},
\]

with \(A_\parallel=n_iA_{ij}n_j\) and \(O_{\rm td}=\tfrac{8}{21}K_{ij}\mathcal D_{ij}(\delta^2-\tfrac32K^2)\)
built from linear fields. The coefficient of \(\delta^2\) is \(b_2/2\).

**Mapped DJS coefficients** (`folps.DJSCoefficients`) include the RSD Jacobian. Ordinary
redshift-space galaxies have \(b_\eta=-1\), \(b_{\delta\eta}=-b_1\), \(b_{\eta^2}=1\) and all six
other selection coefficients zero. The public interface `folps.SelectionBias` holds the nine
**additional** coordinates, zero for no selection:

\[
b_\eta=-1+s_\eta,\quad b_{\delta\eta}=-b_1+s_{\delta\eta},\quad b_{\eta^2}=1+s_{\eta^2},\quad
b_O=s_O\ \ (O=(KK)_\parallel,\Pi^{[2]}_\parallel,\delta\Pi^{[2]}_\parallel,\eta\Pi^{[2]}_\parallel,(K\Pi^{[2]})_\parallel,\Pi^{[3]}_\parallel).
\]

| field | operator | order |
|---|---|---|
| `s_eta` | \(\eta\) | 1 |
| `s_delta_eta`, `s_eta2`, `s_KK_par`, `s_Pi2_par` | \(\delta\eta,\ \eta^2,\ (KK)_\parallel,\ \Pi^{[2]}_\parallel\) | 2 |
| `s_delta_Pi2_par`, `s_eta_Pi2_par`, `s_KPi2_par`, `s_Pi3_par` | \(\delta\Pi^{[2]}_\parallel,\ \eta\Pi^{[2]}_\parallel,\ (K\Pi^{[2]})_\parallel,\ \Pi^{[3]}_\parallel\) | 3 |

The ordinary biases come from the legacy FolpsD vector through the single adapter
`folps.folps_bias_to_djs`: \(b_{K^2}=b_{s^2}/2\), \(b_{\rm td}=-\tfrac{105}{64}b_{3\rm nl}-\tfrac52b_{K^2}\)
(inverse of \(b_{s^2}=2b_{K^2}\), \(b_{3\rm nl}=-\tfrac{32}{21}(b_{K^2}+\tfrac25b_{\rm td})\)). This
normalization was verified against FolpsD's actual loop templates (P22 bias terms to
\(<10^{-5}\), P13 \(\sigma_3^2\) term through the zero-selection comparison; validation report).
Because \(b_{\delta\eta}\) is computed from the current \(b_1\), its \(-b_1\) offset differentiates
with \(b_1\).

**Rest-frame coefficients** (`folps.RestFrameCoefficients`), used by the exact map of
section 4, follow from the exact second/third-order expansion of the redshift-space map
(derived here; verified pointwise for Z2 and, for P13, up to derivative/constant shapes):

\[
b^R_\eta=b_\eta+1=s_\eta,\quad b^R_{\delta\eta}=b_{\delta\eta}+b_1=s_{\delta\eta},\quad
b^R_{\eta^2}=b_{\eta^2}+b_\eta=s_{\eta^2}+s_\eta,\quad
b^R_{\eta\Pi^{[2]}}=b_{\eta\Pi^{[2]}}+b_{\Pi^{[2]}},
\]

all others identical. Thus ordinary RSD is \(b^R_O=0\) for every selection operator.

**Historical tree-only \(b_A\)** (the existing `bA` argument, unchanged):
\(Z_{1,A}=b_{1,F}+f\mu^2+b_A(\mu^2-\tfrac13)=b_{1,D}-b_{\eta,D}f\mu^2\) with
\(b_{1,D}=b_{1,F}-b_A/3\), \(s_\eta=-b_A/f\) (`folps.tree_bA_to_linear_djs`). This is a
linear, constant-\(f\) identity; it fixes no nonlinear coefficient. Passing `bA` together
with `selection` raises an error.

## 3. Kernels through third order (`folps/selection_kernels.py`)

All kernels are *generated* by a small field algebra from the definitions, never copied
from long expressions:

* \(F_n,G_n\): standard unsymmetrized EdS recursion with
  \(\alpha(\mathbf p,\mathbf r)=(\mathbf p+\mathbf r)\cdot\mathbf p/p^2\),
  \(\beta=|\mathbf p+\mathbf r|^2\,\mathbf p\cdot\mathbf r/(2p^2r^2)\), then the normalized permutation average.
* Convective tidal hierarchy \(\Pi^{[1]}=R\delta\),
  \(\Pi^{[a]}=\frac{1}{(a-1)!}[\frac{1}{\mathcal Hf}\frac{D\Pi^{[a-1]}}{D\tau}-(a-1)\Pi^{[a-1]}]\) via the EdS recursion
  \[T^{[a](n)}=\frac{1}{(a-1)!}\Big[(n-a+1)T^{[a-1](n)}-\sum_{m}\mathrm{Sym}\Big(\frac{\mathbf p\cdot\mathbf r}{p^2}G^s_mT^{[a-1](n-m)}\Big)\Big],\quad T^{[1](n)}=R(\mathbf K)F^s_n .\]
* Products, traces, LOS projections, matrix products \((AB)_\parallel\), \(\partial_\parallel\)
  (multiply by \(iK_\parallel\)), full spatial advection \(s^i\partial_i\) (contracted index, *not*
  \(s_\parallel\partial_\parallel\); audit A01) and LOS advection \(u_\parallel\partial_\parallel\).

Independent identities checked (validation report, section 2): closed-form \(F_2,G_2\);
\(\Pi^{[2](2)}=[\Pi^{[1]}\Pi^{[1]}]+\tfrac{10}{21}R(\mathbf K)[\delta^2-\tfrac32K^2]\); continuity
\(\operatorname{tr}\Pi^{[2]}=\vartheta-\delta+\delta\vartheta\) through third order; and, derived here from
the continuity and Euler equations, the third-order trace identity

\[
\operatorname{tr}\Pi^{[3]}=\tfrac74(\delta-\vartheta)+\tfrac12\operatorname{tr}(\Pi_\vartheta\Pi_\vartheta)+\tfrac12\vartheta^2+\tfrac34\delta^2-\tfrac74\delta\vartheta+\tfrac12\delta\vartheta^2+\tfrac12\delta\operatorname{tr}(\Pi_\vartheta\Pi_\vartheta),\qquad \Pi_\vartheta=R\,\vartheta ,
\]

which tests the cubic \(\Pi\) recursion independently of itself.

**Collapsed configurations.** In a 13 contraction the pair \((\mathbf q,-\mathbf q)\) has
exactly zero momentum. Every recursion term in which a zero-momentum block of two or more
momenta enters as a velocity or density sub-kernel multiplied by an inverse power of that
momentum has a vanishing continuous limit after the normalized permutation average (the
two orderings of the pair cancel at first order in the regulator). The oracle sets such
terms to their limit, zero; symmetrized \(F_n,G_n\) and \(T^{[1](n)}\) of a zero-momentum
block vanish, while \(T^{[2](2)}(\mathbf q,-\mathbf q)=R(\mathbf q)\) is finite and kept. Continuity
\(\lim_{\epsilon\to0}Z_3(\mathbf k,\mathbf q,-\mathbf q+\epsilon\hat{\mathbf d})=Z_3(\mathbf k,\mathbf q,-\mathbf q)\) is
tested; no numerical epsilon enters any result.

## 4. Mapped field and exact redshift-space map

**DJS compact mapped field** (equation-note E4):

\[
\delta_{g,s}\big|_{P_{1\ell}}=\sum_{O}b_O[O]-u_\parallel\partial_\parallel\big(b_1\delta+b_\eta\eta+b_{\Pi^{[2]}_\parallel}\Pi^{[2]}_\parallel\big),
\]

all operators with their full nonlinear kernels. **Exact map** of a rest-frame density
\(D\) with comoving galaxies:

\[
\delta_s(\mathbf k)=\int d^3x\,e^{-i\mathbf k\cdot\mathbf x}\big[(1+D)e^{-ik_\parallel u_\parallel}-1\big],
\]
\[
Z_2=D_2-ik_\parallel[U_2+\mathrm{Sym}(D_1U_1)]-\tfrac12k_\parallel^2U_1U_1,\quad
Z_3=D_3-ik_\parallel[U_3+\mathrm{Sym}(D_1U_2)+\mathrm{Sym}(D_2U_1)]-k_\parallel^2[\mathrm{Sym}(U_1U_2)+\tfrac12\mathrm{Sym}(D_1U_1U_1)]+\tfrac i6k_\parallel^3U_1U_1U_1 .
\]

With \(D=\sum_Ob^R_OO\) and the map of section 2, the two fields have **identical** \(Z_1,Z_2\)
(checked at random triangles to \(4\times10^{-15}\)). At third order they differ in the 13
contraction only by derivative (\(k^2\sigma_{-1}^2\)) and constant (\(\sigma_0^2\)) shapes (section
7). The mapped field omits, for example, the double displacement \(\tfrac12u_\parallel^2\partial_\parallel^2D\);
that is why it is "valid for \(P_{1\ell}\) after renormalization" but not as a bare field.

## 5. P22

\[
P_{22}=2\int_{\mathbf q}\Big[Z_2(\mathbf q,\mathbf k-\mathbf q)^2P_L(q)P_L(|\mathbf k-\mathbf q|)-Z_2(\mathbf q,-\mathbf q)^2P_L(q)^2\Big],
\]

i.e. only the isotropic zero-lag constant is removed (absorbed into \(N_0\)); every
\(k\)-dependent piece, including all order-\(k^2\) terms, is kept.

**Ten quadratic kernels** in the equation-note order: \(F_2\), \(-f\mu_K^2G_2\), \(1\),
\(\nu^2-\tfrac13\), \(-f(\mu_1^2+\mu_2^2)/2\), \(f^2\mu_1^2\mu_2^2\),
\(\mu_1\mu_2\nu-(\mu_1^2+\mu_2^2)/3+\tfrac19\), \(\mu_1\mu_2\nu+\tfrac57\mu_K^2(1-\nu^2)\),
\(-f\mu_1\mu_2(r+r^{-1})/2\), \(f^2\mu_1\mu_2(\mu_1^2r+\mu_2^2/r)/2\), with coefficients
\(\mathbf c_2=(b_1,b_\eta,b_2/2,b_{K^2},b_{\delta\eta},b_{\eta^2},b_{(KK)_\parallel},b_{\Pi^{[2]}_\parallel},-b_1,-b_\eta)\).

**Galilean-safe basis (stored).** The two LOS-advection kernels are individually constant
as \(q\to\infty\). The stored tensor instead uses

\[
S'_9=S_9+S_5\ (u_\parallel\partial_\parallel\delta+\delta\eta),\qquad S'_{10}=S_{10}+S_6\ (u_\parallel\partial_\parallel\eta+\eta^2),
\]

both \(O(k^2/q^2)\) at large \(q\) (the uniform-velocity cancellation). Their coefficients are
unchanged and those of \(S_5,S_6\) become \(c'_5=b_{\delta\eta}+b_1=b^R_{\delta\eta}\),
\(c'_6=b_{\eta^2}+b_\eta=b^R_{\eta^2}\): the rest-frame coefficients appear naturally.
Soft-singular kernels (\(\sim1/q\) as \(q\to0\)) are \(\{F_2,\eta^{(2)},S'_9,S'_{10}\}\); all others are
bounded ("local").

**Reduced basis.** Azimuthally averaging \(S'_aS'_b\) about \(\hat{\mathbf k}\) for a general LOS
and writing \(\nu=\hat{\mathbf k}\cdot\hat{\mathbf q}\), every product reduces *exactly* (SymPy rational
arithmetic) to the 23 DJS functions

\[
\mathcal I_{mr}(k)=2\int_{\mathbf q}\frac{q^{r-2}k^{6-r}}{|\mathbf k-\mathbf q|^4}\nu^mP_L(q)P_L(|\mathbf k-\mathbf q|),\qquad
\{(m,r):0\le r\le6,\ m\le r+2,\ r-m\ \text{even}\},
\]

and no other function occurs (asserted by the generator):

\[
P_{22}(k,\mu)=\sum_{a,b}c'_ac'_bf^{n_a+n_b}\sum_{n=0}^{4}\mu^{2n}\sum_{(m,r)}T_{ab,n,mr}\,\mathcal I_{mr}(k)\Big|_{\text{zero-lag removed}},
\qquad n_{a}=(0,1,0,0,1,2,0,0,1,2).
\]

\(T\) (`P22_T`, 1962 nonzero rationals) is symmetric in \(a,b\); the double sum counts
off-diagonal pairs twice.

**UV content.** At order \(k^2\) the large-\(q\) expansion of the selected P22 is
\(k^2(a_0+a_2\mu^2+a_4\mu^4)\) with \(a_4\neq0\) whenever any selection coordinate is nonzero;
\(a_4=0\) without selection, and \(a_6=a_8=0\) always. The validation report quantifies this
and discusses its consequence for the stochastic basis (section 9).

## 6. P13

\[
P_{13}^{\rm total}=6Z_1P_L(k)\int_{\mathbf q}Z_3(\mathbf k,\mathbf q,-\mathbf q)P_L(q)=2Z_1P_L(k)\,R(k,\mu),\qquad R\equiv3\int_{\mathbf q}Z_3P_L .
\]

"Total" includes 13 and 31. \(R\) is the **single** contraction \(\langle\delta_LO^{(3)}\rangle'/P_L\);
DJS Eq. (3.18) is a single contraction and (3.23) multiplies by two. The factor 2 appears
once.

Five reduced integrals, \(Q=|\mathbf k-\mathbf q|\):

\[
J_1=k^2\!\int_{\mathbf q}\frac{\mathbf k\cdot\mathbf q}{q^2Q^2}(1-\nu^2)P_L,\quad
J_2=k^2\!\int_{\mathbf q}\frac{1-\nu^2}{Q^2}P_L,\quad
J_{3,4,5}=\int_{\mathbf q}\Big[\frac{q^2\nu^{0,2,4}(1-\nu^2)}{Q^2}-\tfrac23,\tfrac2{15},\tfrac2{35}\Big]P_L .
\]

**Pointwise identification.** For each reduced operator \(\hat O_a=f^{-n_a}O_a\), the
angular average \(g_a(x,\mu)\) of \(3Z_3^{\hat O_a}(\mathbf k,\mathbf q,-\mathbf q)\) at \(x=q/k\) (polar *and*
azimuthal) satisfies, pointwise in \(x\),

\[
g_a(x,\mu)=\sum_{n=0}^{2}\mu^{2n}\sum_{i=1}^5M^a_{ni}\,j_i(x)+D_a(\mu)\,x^{-2}+C_a(\mu),
\]

with \(j_i\) the angular kernels of \(J_i\). Over-determined fits on 30 values of \(x\)
(avoiding \(x=1\)) give round-off residuals (\(\sim3\times10^{-13}\)), proving that
\(\{j_1..j_5,x^{-2},1\}\) is complete for every operator. The rationalized results:

| # | operator \(O_a\) | \(c_{13,a}\) | \(n_f\) | \(M^a\) rows \((J_1..J_5)\) | bare \(D_a\) | \(C_a\) |
|---|---|---|---|---|---|---|
| 1 | \(\delta^{(3)}\) | \(b_1\) | 0 | \(\mu^0\): (2/21, 1/14, −1/6, 0, 0) | −1/6 | 0 |
| 2 | \(\eta^{(3)}\) | \(b_\eta\) | 1 | \(\mu^2\): (−2/7, 3/14, 1/14, 0, 0) | \(+\mu^2/6\) | 0 |
| 3 | \(2\operatorname{tr}(KK^{(2)})\) | \(b_{K^2}\) | 0 | \(\mu^0\): (10/7, −15/7, 5/7, 0, 0) | 0 | 136/63 |
| 4 | \(\delta\eta^{(2)}\) | \(b_{\delta\eta}\) | 1 | \(\mu^0\): (0, 0, −3/7, 3/7, 0); \(\mu^2\): (0, 0, 9/7, −9/7, 0) | 0 | −8/35 − 58/105 \(\mu^2\) |
| 5 | \(2\eta\eta^{(2)}\) | \(b_{\eta^2}\) | 2 | \(\mu^0\): (0, 0, 9/14, −9/7, 9/14); \(\mu^2\): (6/7, −9/7, −24/7, 72/7, −45/7); \(\mu^4\): (−6/7, 9/7, 39/14, −75/7, 15/2) | 0 | 72/245 + 148/245 \(\mu^2\) |
| 6 | \(2(KK^{(2)})_\parallel\) | \(b_{(KK)_\parallel}\) | 0 | \(\mu^0\): (5/14, −15/28, 5/12, −5/21, 0); \(\mu^2\): (5/14, −15/28, −15/28, 5/7, 0) | 0 | 622/945 + 58/315 \(\mu^2\) |
| 7 | \(O_{\rm td}\) | \(b_{\rm td}\) | 0 | \(\mu^0\): (4/7, −6/7, 2/7, 0, 0) | 0 | 32/63 |
| 8 | \(\delta\Pi^{[2]}_\parallel\) | \(b_{\delta\Pi^{[2]}_\parallel}\) | 0 | \(\mu^0\): (0, 0, 5/7, −5/7, 0); \(\mu^2\): (0, 0, −15/7, 15/7, 0) | 0 | 5/7 + 10/21 \(\mu^2\) |
| 9 | \(\eta\Pi^{[2]}_\parallel\) | \(b_{\eta\Pi^{[2]}_\parallel}\) | 1 | \(\mu^0\): (0, 0, −15/28, 15/14, −15/28); \(\mu^2\): (−5/7, 15/14, 20/7, −60/7, 75/14); \(\mu^4\): (5/7, −15/14, −65/28, 125/14, −25/4) | 0 | −12/49 − 419/735 \(\mu^2\) |
| 10 | \((K\Pi^{[2]})_\parallel\) | \(b_{(K\Pi^{[2]})_\parallel}\) | 0 | \(\mu^0\): (5/28, −15/56, 5/24, −5/42, 0); \(\mu^2\): (5/28, −15/56, −15/56, 5/14, 0) | 0 | 22/105 + 142/315 \(\mu^2\) |
| 11 | \(s^i\partial_i\Pi^{[2]}_\parallel\) | \(-b_{\Pi^{[2]}_\parallel}\) | 0 | \(\mu^0\): (5/28, −15/56, 25/56, −5/14, 0); \(\mu^2\): (−15/28, 45/56, −75/56, 15/14, 0) | 0 | 8/21 + 10/21 \(\mu^2\) |
| 12 | \(u^{(2)}_\parallel\partial_\parallel\delta\) | \(-b_1\) | 1 | \(\mu^0\): (0, 0, 3/7, −3/7, 0); \(\mu^2\): (0, −3/7, −6/7, 9/7, 0) | 0 | 8/35 + 23/105 \(\mu^2\) |
| 13 | \(u^{(2)}_\parallel\partial_\parallel\eta\) | \(-b_\eta\) | 2 | \(\mu^0\): (0, 0, −9/28, 9/14, −9/28); \(\mu^2\): (−9/28, 27/56, 9/8, −9/2, 45/14); \(\mu^4\): (15/28, −3/8, −39/56, 30/7, −15/4) | 0 | −36/245 − 5/49 \(\mu^2\) |
| 14 | \(u_\parallel\partial_\parallel\eta^{(2)}\) | \(-b_\eta\) | 2 | \(\mu^0\): (0, 0, −9/28, 9/14, −9/28); \(\mu^2\): (−9/28, 27/56, 135/56, −81/14, 45/14); \(\mu^4\): (15/28, −45/56, −135/56, 45/7, −15/4) | \(-\mu^4/3\) | −36/245 − 123/245 \(\mu^2\) |
| 15 | \(u_\parallel\partial_\parallel\Pi^{[2]}_\parallel\) | \(-b_{\Pi^{[2]}_\parallel}\) | 1 | \(\mu^0\): (0, 0, 15/28, −15/14, 15/28); \(\mu^2\): (15/28, −45/56, −225/56, 135/14, −75/14); \(\mu^4\): (−25/28, 75/56, 225/56, −75/7, 25/4) | 0 | 12/49 + 58/245 \(\mu^2\) |
| 16 | \(\Pi^{[3]}_\parallel\) | \(b_{\Pi^{[3]}_\parallel}+2b_{\Pi^{[2]}_\parallel}\) | 0 | \(\mu^0\): (13/56, −39/112, 65/112, −13/28, 0); \(\mu^2\): (−101/168, 101/112, −569/336, 39/28, 0) | 0 | 59/105 + 58/105 \(\mu^2\) |

All sixteen agree exactly with GeneralBiasPk `P13/M13.jl` (commit `d4d5bee`). Rows 11 and 16
encode \([\Pi^{[2]}_\parallel]^{(3)}=2\Pi^{[3](3)}_\parallel-s^i\partial_i\Pi^{[2]}_\parallel\), which follows from the
\(T\) recursion (\(T^{[2](3)}=2T^{[3](3)}+\mathrm{Sym}[(\mathbf p\cdot\mathbf r/p^2)G_1T^{[2](2)}]\)) and confirms the
full spatial advection.

**The \(\eta^{(3)}\) row (audit A02).** Adopted:
\(M[f^{-1}\eta^{(3)}]_{\mu^2}=\tfrac17(-2,\tfrac32,\tfrac12,0,0)\), equivalently
\(3\int_{\mathbf q}G_3P_L=\tfrac27J_1-\tfrac3{14}J_2-\tfrac1{14}J_3-\tfrac16k^2\sigma_{-1}^2\) with
\(\eta^{(3)}=-f\mu^2G_3\). The printed DJS Appendix F.7 row \((2,\tfrac32,-\tfrac12,0,0)/7\) is
inconsistent with this convention and with the paper's own Appendix E.9. GeneralBiasPk uses
the adopted row. No author erratum was located; see the sources document.

**UV pieces of the continued \(J\)'s.** FFTLog evaluates the analytically continued
integrals, which drop the power-divergent \(k^2\sigma_{-1}^2\) pieces:
\(J_i^{\rm true}=J_i^{\rm cont}+\delta J_i\,k^2\sigma_{-1}^2\) with
\(\delta J=(0,\tfrac23,-\tfrac2{15},\tfrac2{21},\tfrac{22}{315})\) (large-\(x\) expansion of \(j_i\), SymPy).
As a consistency check, \(2[M^{\delta^{(3)}}\!\cdot\delta J-\tfrac16]=-\tfrac{61}{315}\) reproduces
FolpsD's \(-\tfrac{61}{105}k^2\sigma_\psi^2\) in `P13dd`, and the \(\vartheta\) analogue its
\(-\tfrac95k^2\sigma_\psi^2\) in `P13tt`.

## 7. Renormalization bridge and the final P13

Applying the same pointwise identification to the full mapped field (A) per DJS
coefficient, and to the exact map (B) per rest-frame coefficient (and its matter part), as
polynomials in \(f\) and \(\mu^2\):

1. The \(J\) parts of (A) equal \(\sum_ac_{13,a}f^{n_a}M^a\) with \(\mathbf c_{13}\) of the table;
   the \(J\) parts of (B) equal the same under the coefficient map of section 2 (max
   deviation at fit-noise level, validation report).
2. (A) and (B) differ only in \(x^{-2}\) and constant coefficients.
3. **Bare derivative term of the exact map, in closed form:**
   \[
   D_B(\mu)=-\tfrac16\big[1+f\mu^2(2+f)\big]\,Z_1,\qquad Z_1=b_1-b_\eta f\mu^2 ,
   \]
   with *zero* contribution from every quadratic and cubic selection coefficient. Hence
   \[
   P_{13}\supset2Z_1P_L\,k^2\sigma_{-1}^2D_B=-k^2\sigma_\psi^2\big[1+f\mu^2(2+f)\big]Z_1^2P_L ,
   \]
   the long-wavelength (Zel'dovich) displacement damping. It uses the physical \(f\) and
   the *selected* linear kernel: selection does not change the displacement variance.
4. The constants \(C\) (table above and `BRIDGE_C`) renormalize \(b_1\) and \(b_\eta\) and are
   not included, as in FolpsD (e.g. the \(b_2\) term \(\tfrac{34}{21}+\tfrac12f\mu^2\) per unit
   \(b_2\sigma_0^2\)).

**Implemented P13 (bare scheme, FolpsD-compatible):**

\[
P_{13}=2Z_1P_L\Big\{\sum_{a=1}^{16}c_{13,a}f^{n_a}(1,\mu^2,\mu^4)M^a\cdot\big(\mathbf J^{\rm cont}+\delta\mathbf J\,k^2\sigma_{-1}^2\big)
-\tfrac16\big[1+f\mu^2(2+f)\big]Z_1\,k^2\sigma_{-1}^2\Big\}.
\]

**Bridge to the DJS reduced convention** (equation-note E8). With
\(L_{\rm reduced}=P_{22}+2Z_1P_L\sum_ac_{13,a}f^{n_a}(1,\mu^2,\mu^4)M^a\cdot\mathbf J^{\rm true}\),

\[
L_{\rm FolpsD\text{-}bare}-L_{\rm reduced}=k^2P_L\sum_{r=0}^{3}d_{2r}\mu^{2r},\qquad
\sum_rd_{2r}\mu^{2r}=-\sigma_\psi^2\big[1+f\mu^2(2+f)\big]Z_1^2 ,
\]

which is exactly of the derivative-counterterm shape \(k^2P_L\mu^{0..6}\), fixed by algebra
(no fit), and IR-relevant. Using (A)'s own bare \(x^{-2}\) terms instead would break the IR
cancellation, because the mapped field lacks the double displacement. The kernel-level
soft limit \(\lim_{q\to0}q^2[4\langle Z_2^2\rangle+6Z_1\langle Z_3\rangle]\) vanishes for (B) as \((q/k)^2\)
and stays \(O(1)\) for (A).

At zero selection the implemented loop reproduces FolpsD's complete one-loop EdS EFT loop
(Kaiser loops, A/B (D) terms and GTNS) in a common scheme (validation report).

## 8. Assembly with the FolpsD baseline

At fixed \(b_1,b_2,b_{K^2},b_{\rm td}\), with \(\mathbf b_0\) the same biases with \(\mathbf s=0\) and the same
\(P_L,f\):

\[
\Delta L=L(\mathbf b)-L(\mathbf b_0),\qquad \Delta T=(Z_1^2-Z_{1,0}^2)P_L ,
\]

computed for the full-shape and no-wiggle tables separately. The increments are
evaluated as exact differences of coefficient vectors (\(cc^T-c_0c_0^T=\Delta c\,c^T+c_0\Delta c^T\)),
so they are identically zero for \(\mathbf s=0\). The explicit zero `SelectionBias()` runs the
full machinery and returns the legacy prediction bit for bit. GTNS is already part of
FolpsD's complete loop, and the increment carries its selected response. No second
GTNS-like term is added, and \(f\) is never replaced by \(f+b_A\).

## 9. Derivative and stochastic sectors

Legacy counterterms \((\alpha_0+\alpha_2\mu^2+\alpha_4\mu^4)k^2P_L\) and noise
\(P_{\rm shot}(\alpha_{\rm shot,0}+\alpha_{\rm shot,2}k^2\mu^2)\) are unchanged. `folps.EFTExtension` adds

* `alpha6`: \(\alpha_6\mu^6k^2P_L\) (IR-mixed like the other counterterms), \([\alpha_6]=L^2\);
* `alphashot2_iso`: \(P_{\rm shot}\alpha_{\rm shot,2}^{\rm iso}k^2\) (\(N_{2,\rm iso}\)), \([\,]=L^2\);
* `alphashot2_mu4`: \(P_{\rm shot}\alpha_{\rm shot,2}^{\mu^4}k^2\mu^4\), a *proposed* term (not in the DJS
  basis) motivated by the \(k^2\mu^4\) UV content of the selected P22 (section 5).

Zero values reproduce the legacy model. There is no anisotropic white noise \(\mu^2N_0\).

`folps.PhysicalDerivativeBias(b_nabla2delta, beta_nabla2v, beta_par2v)` *replaces* the
polynomial sector,
\(P_{\rm hd}=-2k^2P_LZ_1[b_{\nabla^2\delta}-b_\eta f\mu^2(\beta_{\nabla^2v}+\beta_{\partial_\parallel^2v}\mu^2)]\), i.e.
\((c_0,c_2,c_4,c_6)=(-2b_1d,\,2a(d+b_1v),\,2b_1aw-2a^2v,\,-2a^2w)\), \(a=b_\eta f\). Nonzero legacy
\(\alpha\)'s together with this mode raise an error. The four-coefficient polynomial basis is
broader than the three physical coefficients (\(c_6\) is fixed by the others when
\(b_1a\neq0\)). Whether \(\mu^6\) adds a direction depends on the projection: for \(\ell\le4\),
unity AP and no IR, it is exactly degenerate with \(\alpha_{0,2,4}\) (notebook section 14).

`ctilde` (a \(k^4P_L\)-type higher-order term) keeps its legacy definition. In selection
mode its Kaiser factor is the selected \(Z_1^2\) (added as an increment). Set `ctilde=0` for
strict one-loop work.

## 10. FFTLog construction

With FolpsD's decomposition \(P_L(k)=\sum_mc_mk^{-2\nu_m}\) (`get_cm`, \(N\) = `nfftlog`,
\(k\in[10^{-7},100]\)) every term reduces to

\[
\int_{\mathbf q}q^{-2a}|\mathbf k-\mathbf q|^{-2b}=\frac{k^{3-2a-2b}}{8\pi^{3/2}}\frac{\Gamma(\tfrac32-a)\Gamma(\tfrac32-b)\Gamma(a+b-\tfrac32)}{\Gamma(a)\Gamma(b)\Gamma(3-a-b)}
\]

after expanding \(\nu^m=(k^2+q^2-Q^2)^m/(2kq)^m\), and so to `K^3 vec @ M @ vec` (P22) or
`K^3 vec @ M` (P13), exactly as in `MatrixCalculator`. Matrices are cosmology independent
(`folps.get_selection_matrices`, complex128, cached by a key containing generator version,
coefficient checksum, grid, tilt and dtype; incompatible caches are rejected). The \(J_n\)
use FolpsD's matter tilt \(b=-0.1\); scaleless (\(Q^{\ge0}\)) monomials vanish.

**P22 tilts.** The continued integral at tilt \(b\) drops the UV pieces that are power-divergent
for \(q^{b}\) and keeps the convergent ones. Two classes:

* soft-singular × soft-singular pairs are \(O(k^4/q^4)\) in the UV (no constant, no \(k^2\)):
  tilt \(b=-0.1\), IR-convergent for their \(1/q^2\) soft behavior. There they equal the true
  integral.
* all other pairs (at most \(1/q\) soft) at `P22_TILT_LOCAL` \(=-0.9\). For \(b\in(-\tfrac32,-\tfrac12)\) the
  continued integral drops **exactly the zero-lag constant** and keeps the order-\(k^2\) pieces,
  i.e. it is the definition of section 5 with no explicit subtraction or restoration.

The generator asserts the classification in exact arithmetic. Within the strip, the
periodic images of the FFTLog representation leak into the \(k^2\) and constant pieces with
weights \(\sim(k_{\max}/k_{\min})^{1+2b}\) and \((k_{\max}/k_{\min})^{-(3+2b)}\). Hence \(-0.9\): \(b=-1\) itself
hits a spurious pole of the reduced basis at \(\mathrm{Re}\,\nu=\tfrac12\). (Restoring dropped
\(k^2\) pieces analytically at \(b=-0.1\) is correct but ill-conditioned, since the restored
piece is several times the answer, so it is not used.)

## 11. IR resummation, AP and projection

Following the actual FolpsD mixing, with \(P_n=P_{\rm nw}\), \(P_w=P_L-P_n\),
\(A=k^2\Sigma_s^2(\mu)\), \(E=e^{-A}\), \(\Sigma_s^2=[1+f\mu^2(2+f)]\Sigma^2+f^2\mu^2(\mu^2-1)\Delta\Sigma^2\):

\[
P_{\rm IR}=Z_1^2\{P_n+E(1+A)P_w\}+L[P_n]+E\{L[P_L]-L[P_n]\}+C[P_n]+E\{C[P_L]-C[P_n]\}+P_{\rm stoch}.
\]

Selection enters \(Z_1\), \(L\) and \(C\) (physical-derivative mode) but **not** \(\Sigma_s\). The wiggle
loop is \(L[P_L]-L[P_n]\) with both tables built from the same calculator call (enforced via
a shared table identifier), never \(L[P_w]\). AP: \(k=k_{\rm obs}[\mu_{\rm obs}^2/q_\parallel^2+(1-\mu_{\rm obs}^2)/q_\perp^2]^{1/2}\),
\(\mu=(\mu_{\rm obs}/q_\parallel)/[\cdots]^{1/2}\), \(P_{\rm obs}=P(k,\mu)/(q_\parallel q_\perp^2)\), Legendre projection with
FolpsD's symmetric Gauss rule (`nmu` nodes on \([0,1]\), i.e. \(2\,\)`nmu` full-range nodes). The
unresummed unity-AP deterministic polynomial stops at \(\mu^8\); with IR or AP it does not.
The window is applied in the downstream pipeline.

## 12. Data schema, regeneration and limits

`folps/_selection_coefficients_data.py` (generated; exact `(numerator, denominator)` pairs):

| name | meaning |
|---|---|
| `QUADRATIC_KERNELS`, `QUADRATIC_NF`, `QUADRATIC_SINGULAR` | Galilean-safe quadratic basis, growth powers, soft-singular set |
| `P22_BASIS` | the 23 \((m,r)\) |
| `P22_TENSOR[(a,b,n,m,r)]` | \(T_{ab,n,mr}\), \(a\le b\) |
| `P22_UV_K2[(m,r)]` | order-\(k^2\) UV coefficients \((a_{R_0},a_S)\): \(k^2(a_{R_0}R_0+a_SS)/2\pi^2\), \(R_0=\int P^2dq\), \(S=\int q^2P'^2dq\) (diagnostics) |
| `J_UV` | \(\delta J\) |
| `CUBIC_OPERATORS`, `CUBIC_NF`, `CUBIC_M`, `CUBIC_D`, `CUBIC_C` | table of section 6 |
| `REST_KEYS`, `BRIDGE_D`, `BRIDGE_C` | per rest-frame coefficient \(x^{-2}\) and constant polynomials \([f^p,\mu^{2n}]\) of the exact map |

Regenerate (≈1 h, one core; SymPy for P22, oracle identification for P13):

```bash
python scripts/generate_selection_coefficients.py --resume-dir /tmp/selection_gen
python scripts/generate_selection_coefficients.py --check     # regenerate and diff
```

The report `docs/selection_1loop_generation_report.json` lists every residual.

**Not implemented.** Selected cross spectra (rejected); `fk` / scale-dependent growth
selection kernels (rejected: rederive the velocity kernels with \(f(k)\), convective time
derivatives of \(\Pi^{[a]}\) with non-separable growth, the operator-basis equivalences, which
are EdS identities, and all templates; a global \(f\to f(k)\) is not a valid substitute);
FoG-damped models with selection (rejected); bispectrum and `folps_bins_beta.py`
(untouched).
