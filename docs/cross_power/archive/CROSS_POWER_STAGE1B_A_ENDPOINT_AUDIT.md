# Cross-Power Stage 1B A-Endpoint Audit

This is an audit and derivation note only. I read the TeX theory note,
`docs/cross_power/CROSS_POWER_STAGE1_AUDIT.md`, current source needed for the
matrix and loop-table mapping, and Git history. I did not read the PDF, did not
modify production source, did not modify tests, did not generate production
matrices, and did not commit or push.

External references checked for definitions only:

- Taruya, Nishimichi, and Saito, TNS redshift-space model,
  arXiv:1006.0699 / Phys. Rev. D 82, 063522.
- Aviles et al., FOLPS conventions, arXiv:2106.13771.
- Noriega et al., FFTLog FOLPS implementation, arXiv:2208.02791.
- Bansal et al., FolpsD, arXiv:2604.08895.

## Executive Conclusion

The ordered endpoint exchange does not require new analytic FFTLog matrices.
The missing point from Stage 1 is that endpoint exchange swaps both the matrix
arguments and the two FFTLog coefficient vectors:

```text
endpoint exchange:  x^T M y  ->  y^T M^T x
```

The exchanged object is therefore not `x^T M^T y`. For the scalar bilinear used
by the loop table,

```text
x^T M y = y^T M^T x
```

algebraically, even when `x != y`, provided the vectors are exchanged with the
two endpoints. This resolves the `vecf_b @ MtAfp_11_b2 * vec_b` and analogous
`bs2` concern: the exchanged endpoint is `vec_b @ MtAfp_11_b2.T * vecf_b`, not
`vecf_b @ MtAfp_11_b2.T * vec_b`.

The full cross-A term can be built from existing scalar rows by polarizing the
endpoint-summed auto rows. No additional production scalar rows are required.

## Definitions and Bias Convention

The TNS/FOLPS A term is

```math
A_{AB}(k,\mu)
= k\mu f_0 \int_p {p_\parallel \over p^2}
\left[
B_{\sigma;AB}(\mathbf p,\mathbf k-\mathbf p,-\mathbf k)
-B_{\sigma;AB}(\mathbf p,\mathbf k,-\mathbf k-\mathbf p)
\right],
```

with `int_p = int d^3p/(2 pi)^3`. The ordered bispectrum is

```math
(2\pi)^3\delta_D(\mathbf k_1+\mathbf k_2+\mathbf k_3)
B_{\sigma;AB}(\mathbf k_1,\mathbf k_2,\mathbf k_3)
=
\left<\theta(\mathbf k_1)\mathcal S_A(\mathbf k_2)
\mathcal S_B(\mathbf k_3)\right>.
```

The source is

```math
\mathcal S_X(\mathbf q)
= \delta_X(\mathbf q)+f_0\mu_q^2\theta(\mathbf q).
```

For the endpoint audit it is useful to split

```math
D_X=b_1^X\delta,\qquad
U=f_0\mu_q^2\theta,\qquad
Q_X={b_2^X\over2}[\delta^2],\qquad
T_X={b_{s^2}^X\over2}[s^2]_{\rm code}.
```

The `1/2` multiplying `b2` is the usual local-quadratic convention. The `1/2`
multiplying canonical FolpsD `bs2` is the current code convention: the
bispectrum kernels use

```python
term1 = b2/2 + bs/2 * (xij**2 - 1/3)
```

at `folps/folps.py:2266-2269` and `folps/folps.py:3183-3185`, while
`classpt`/`DESI` input schemes map `bs2 = 2*bG2` or `bs2 = 2*bK2` at
`folps/folps.py:1377-1391`. Thus the cross formulas below are in canonical
FolpsD parameters.

Define the linear spectra and FFTLog vectors:

```math
F(q)=f(q)/f_0,\qquad
P_d(q)=P_L(q),\qquad
P_u(q)=F(q)P_L(q),\qquad
P_{uu}(q)=F(q)^2P_L(q).
```

Code vectors:

```text
vec    = cmT   * precvec      -> P_d
vecf   = cmTf  * precvec      -> P_u
vecff  = cmTff * precvec      -> P_uu
vec_b  = cmT_b  * precvec_b   -> P_d with FFTLog bias bnu_b
vecf_b = cmTf_b * precvec_b   -> P_u with FFTLog bias bnu_b
```

The suffix `_b` means the auxiliary FFTLog bias exponent, not tracer `B`.
The setup is at `folps/folps.py:1016-1041` and `folps/folps.py:1076-1081`.

## Tree-Level Channel Origin

The ordered tree-level contributions are:

| Channel | Tree-level source products | Cross coefficient after endpoint polarization |
|---|---|---|
| `I1udd_1` | `<theta, D_A, D_B>` | `b1A*b1B*f0*mu^2` |
| `I2uud_1`, `I2uud_2` | `<theta, U, D_B> + <theta, D_A, U>` | `(b1A+b1B)/2 * f0^2 * (mu^2 I2uud_1 + mu^4 I2uud_2)` |
| `I3uuu_2`, `I3uuu_3` | `<theta, U, U>` | `f0^3 * (mu^4 I3uuu_2 + mu^6 I3uuu_3)` |
| `I1udd_1_b2` | `<theta, Q_A, D_B> + <theta, D_A, Q_B>` | `(b2A*b1B+b1A*b2B)/4 * f0*mu^2` |
| `I2uud_1_b2`, `I2uud_2_b2` | `<theta, Q_A, U> + <theta, U, Q_B>` | `(b2A+b2B)/4 * f0^2 * (mu^2 I2uud_1_b2 + mu^4 I2uud_2_b2)` |
| `I1udd_1_bs2` | `<theta, T_A, D_B> + <theta, D_A, T_B>` | `(bs2A*b1B+b1A*bs2B)/4 * f0*mu^2` |
| `I2uud_1_bs2`, `I2uud_2_bs2` | `<theta, T_A, U> + <theta, U, T_B>` | `(bs2A+bs2B)/4 * f0^2 * (mu^2 I2uud_1_bs2 + mu^4 I2uud_2_bs2)` |

For `b2`, the two Wick contractions inside `[delta^2]` are part of the matrix
kernel normalization; the explicit source coefficient is still `b2/2`. For
canonical `bs2`, the explicit source coefficient is also `bs2/2` in current
FolpsD.

## Endpoint Exchange

Start from configuration space:

```math
A_{AB}=-ik_\parallel f_0\int d^3r\,e^{-i\mathbf k\cdot\mathbf r}
\left< [u_\parallel(\mathbf x_2)-u_\parallel(\mathbf x_1)]
\mathcal S_A(\mathbf x_1)\mathcal S_B(\mathbf x_2)\right>_c .
```

Exchange endpoints with `r'=-r`:

```text
x1 <-> x2,
r -> -r,
Delta u -> -Delta u,
exp(-i k.r) -> exp(+i k.r') = exp(-i (-k).r'),
k_parallel -> -k_parallel when the result is written at -k.
```

The sign from `Delta u` is canceled by the sign from `k_parallel`, and the
equal-time parity-even spectrum satisfies `A_YX(-k,-mu)=A_YX(k,mu)`. Therefore
the exchanged endpoint has the same scalar value and no extra minus sign.

For all P22-type FFTLog reductions, write the stored scalar as

```text
S[x, M, y] = K^3 * sum_i sum_j x_i M_ij y_j.
```

`M22type` builds `M_ij = Mfunc(nu_i, nu_j)` through
`M22(nuT_y, nuT_x)` at `folps/folps.py:777-791`. Exchanging the two source
endpoints is the loop-variable change

```text
p <-> k-p,
mu_p <-> mu_{k-p},
kernel(nu_i, nu_j) -> kernel(nu_j, nu_i),
left spectrum <-> right spectrum.
```

Thus

```text
S[x, M, y] -> S[y, M.T, x] = K^3 * sum_i sum_j y_i M_ji x_j = S[x, M, y].
```

This is the key distinction:

```text
x^T M y        original ordered scalar
x^T M^T y      transpose without endpoint-vector exchange; generally different
y^T M^T x      actual exchanged endpoint; identical scalar
```

The equality is bilinear, not Hermitian; no complex conjugation is involved.
The code takes `.real` after the FFTLog sum, so the equality is preserved.

For P13-type pieces, there is only one loop momentum. Endpoint exchange maps
the external-`k` weighted vector piece into the internal-`p` weighted vector
piece. The current rows already combine those paired terms, for example
`I1udd_1a` combines `Fkoverf0[:, None] * vec @ Mafk_11` and `vecf @ Mafp_11`
at `folps/folps.py:1182-1185`.

## Code Mapping and Endpoint Status

The original A matrices are defined at `folps/folps.py:565-581`, P22
contractions at `folps/folps.py:1097-1102`, P13 contractions at
`folps/folps.py:1182-1185`, and final row combinations at
`folps/folps.py:1265-1269`.

| Channel | Loop-table variable | Matrix / row pieces | Left vector | Right vector | Source lines | Exchanged endpoint candidate | Stored scalar status | Classification |
|---|---|---|---|---|---|---|---|---|
| `I1udd_1` | `I1udd_1` | P22 `MtAfp_11`; P13 `Mafk_11`, `Mafp_11` | P22 `vecf`; P13 `Fk*vec`, `vecf` | P22 `vec`; P13 none | matrices `565-566`, `745-749`; contractions `1098`, `1183`; combine `1265` | P22 `vec @ MtAfp_11.T * vecf`; P13 two subterms exchange | endpoint-symmetric tree sum | `REUSE_SCALAR_ROW` |
| `I2uud_1` | `I2uud_1` | P22 `MtAfkmpfp_12`; P13 `Mafkfp_12`, `Mafpfp_12` | P22 `vecf`; P13 `Fk*vecf`, `vecff` | P22 `vecf`; P13 none | matrices `568-569`, `751-755`; contractions `1099`, `1184`; combine `1266` | P22 `vecf @ MtAfkmpfp_12.T * vecf`; P13 subterms exchange | sum of density-at-A and density-at-B endpoint orderings | `POLARIZE_EXISTING_ROW` |
| `I2uud_2` | `I2uud_2` | P22 `MtAfkmpfp_22` plus `MtAfpfp_22`; P13 pieces from `I3uuu_3a` and `I1udd_1a` | P22 `vecf`, `vecff`; P13 inherited | P22 `vecf`, `vec`; P13 none | matrices `571-575`, `745-761`; contractions `1101`, `1183-1185`; combine `1267` | `vecf @ MtAfkmpfp_22.T * vecf` and `vec @ MtAfpfp_22.T * vecff` | sum of density-at-A and density-at-B endpoint orderings | `POLARIZE_EXISTING_ROW` |
| `I3uuu_2` | `I3uuu_2` | P22 `MtAfkmpfpfp_23`; P13 from `I2uud_1a` | P22 `vecff`; P13 inherited | P22 `vecf`; P13 none | matrix `577-578`; contractions `1102`, `1184`; combine `1268` | `vecf @ MtAfkmpfpfp_23.T * vecff` | velocity endpoints only; endpoint-symmetric | `REUSE_SCALAR_ROW` |
| `I3uuu_3` | `I3uuu_3` | P22 `MtAfkmpfpfp_33`; P13 `Mafkfkfp_33`, `Mafkfpfp_33` | P22 `vecff`; P13 `Fk*vecf`, `vecff` | P22 `vecf`; P13 none | matrices `580-581`, `757-761`; contractions `1100`, `1185`; combine `1269` | `vecf @ MtAfkmpfpfp_33.T * vecff` | velocity endpoints only; endpoint-symmetric | `REUSE_SCALAR_ROW` |

The full biased rows have no P13 piece in the current code. Their matrices are
defined at `folps/folps.py:648-653` and `folps/folps.py:705-716`, contracted at
`folps/folps.py:1132-1146`, tabled at `folps/folps.py:1289-1297`, unpacked at
`folps/folps.py:1460-1466`, and scaled at `folps/folps.py:1492-1507`.

| Channel | Loop-table variable | Matrix name | Left vector | Right vector | Exact source line range | Exchanged endpoint candidate | Stored scalar status | Classification |
|---|---|---|---|---|---|---|---|---|
| `I1udd_1_b2` | `I1udd_1_b2` | `MtAfp_11_b2` | `vecf_b` | `vec_b` | matrix `706-707`; contraction `1134`; table `1291,1295`; pkmu `1465,1492-1507` | `vec_b @ MtAfp_11_b2.T * vecf_b` | endpoint sum after ordered exchange; not labelled in code, identified by derivation and auto normalization | `POLARIZE_EXISTING_ROW` |
| `I2uud_1_b2` | `I2uud_1_b2` | `MtAfkmpfp_12_b2` | `vecf_b` | `vecf_b` | matrix `712-713`; contraction `1135`; table `1292,1295`; pkmu `1465,1492-1507` | `vecf_b @ MtAfkmpfp_12_b2.T * vecf_b` | endpoint sum | `POLARIZE_EXISTING_ROW` |
| `I2uud_2_b2` | `I2uud_2_b2` | `MtAfkmpfp_22_b2` | `vecf` | `vecf` | matrix `649-650`; contraction `1136`; table `1293,1295`; pkmu `1465,1492-1507` | `vecf @ MtAfkmpfp_22_b2.T * vecf` | endpoint sum | `POLARIZE_EXISTING_ROW` |
| `I1udd_1_bs2` | `I1udd_1_bs2` | `MtAfp_11_bs2` | `vecf_b` | `vec_b` | matrix `709-710`; contraction `1138`; table `1291,1296`; pkmu `1465,1495-1507` | `vec_b @ MtAfp_11_bs2.T * vecf_b` | endpoint sum after ordered exchange; not labelled in code, identified by derivation and auto normalization | `POLARIZE_EXISTING_ROW` |
| `I2uud_1_bs2` | `I2uud_1_bs2` | `MtAfkmpfp_12_bs2` | `vecf_b` | `vecf_b` | matrix `715-716`; contraction `1139`; table `1292,1296`; pkmu `1465,1495-1507` | `vecf_b @ MtAfkmpfp_12_bs2.T * vecf_b` | endpoint sum | `POLARIZE_EXISTING_ROW` |
| `I2uud_2_bs2` | `I2uud_2_bs2` | `MtAfkmpfp_22_bs2` | `vecf` | `vecf` | matrix `652-653`; contraction `1140`; table `1293,1296`; pkmu `1465,1495-1507` | `vecf @ MtAfkmpfp_22_bs2.T * vecf` | endpoint sum | `POLARIZE_EXISTING_ROW` |

For the two mixed-vector rows, the important equality is explicitly

```text
sum(vecf_b @ M * vec_b)
= sum(vec_b @ M.T * vecf_b)
!= generally sum(vecf_b @ M.T * vec_b).
```

## Cross-Tracer A Polynomials

With `A_full=False`, the exact cross-tracer A polynomial is

```math
A_{AB}^{A_{\rm full}=False}
= b_1^A b_1^B f_0\mu^2 I1udd_1
+ {b_1^A+b_1^B\over2} f_0^2
  \left(\mu^2 I2uud_1+\mu^4 I2uud_2\right)
+ f_0^3
  \left(\mu^4 I3uuu_2+\mu^6 I3uuu_3\right).
```

With `A_full=True`, add

```math
\Delta A_{AB}^{b_2}
= {b_2^A b_1^B+b_1^A b_2^B\over4}
   f_0\mu^2 I1udd_1_b2
+ {b_2^A+b_2^B\over4} f_0^2
   \left(\mu^2 I2uud_1_b2+\mu^4 I2uud_2_b2\right),
```

and

```math
\Delta A_{AB}^{b_{s^2}}
= {b_{s^2}^A b_1^B+b_1^A b_{s^2}^B\over4}
   f_0\mu^2 I1udd_1_bs2
+ {b_{s^2}^A+b_{s^2}^B\over4} f_0^2
   \left(\mu^2 I2uud_1_bs2+\mu^4 I2uud_2_bs2\right).
```

The factors of `1/4` are `1/2` from endpoint polarization of an endpoint-summed
row times `1/2` from the current second-order source convention for canonical
`b2` or `bs2`. In the auto limit `A=B`, these reduce exactly to the implemented
auto coefficients:

```text
(b1*b2*f0/2) * mu^2 * I1udd_1_b2
+ (b2*f0^2/2) * (mu^2 * I2uud_1_b2 + mu^4 * I2uud_2_b2)
+ (b1*bs2*f0/2) * mu^2 * I1udd_1_bs2
+ (bs2*f0^2/2) * (mu^2 * I2uud_1_bs2 + mu^4 * I2uud_2_bs2)
```

as shown in `folps/folps.py:1506-1507`.

## Answers to Required Questions

1. The complete equal-time cross A term can be implemented without new analytic
   FFTLog matrices.
2. No additional production scalar contractions or loop-table rows are needed.
   The existing rows must be polarized with the coefficients above. Diagnostic
   scripts may still compare `x^T M y`, `y^T M^T x`, `x^T M^T y`, and
   `y^T M x`, but this is not required for the analytic conclusion.
3. For `A_full=False`, use the polynomial in the previous section.
4. For `A_full=True`, use the `A_full=False` polynomial plus the `b2` and
   `bs2` additions above.
5. The current canonical FolpsD convention inserts both `b2` and `bs2` into the
   second-order source with an explicit `1/2`; endpoint polarization of stored
   endpoint-summed rows supplies the additional `1/2` in cross mode.
6. Each current full-A row represents the endpoint-summed auto row after the
   ordered endpoint exchange. The source code does not name it that way, but
   the exchange derivation and the auto normalization identify it as the sum,
   not as a separately exposed one-endpoint scalar.

## Git History

Pickaxe searches used `A_full`, `MtAfp_11_b2`, `MtAfkmpfp_22_b2`,
`I1udd_1_b2`, and the comment text `A function: contributions due to b2 & bs2`.

Relevant current-branch history:

| Commit | Files | Relevant content | Endpoint symmetrized? |
|---|---|---|---|
| `851e8495d1975f86c6e43497f4bcb347e9c958f6` (`All tests passed`, 2026-03-22) | Adds current `folps/folps.py`, docs, tests, and matrix file | Introduces `A_full`, the six full-A rows, matrix formulas, contractions, and the `/2` scaling used today | No derivation or endpoint-symmetry statement found; comments only say `A function: contributions due to b2 & bs2` |
| `10948da5bc55bc06fc907302dfca090497227f87` (`Add files via upload`, 2026-06-15) | Adds `folps/folps_bins_beta.py` | Carries the same full-A formulas in the binned beta variant | No endpoint derivation found |

Older non-ancestor or side-branch history also contains copies of the same
formulas:

| Commit | Files | Relevant content | Endpoint symmetrized? |
|---|---|---|---|
| `8b2b06944712df30eb747e2e8bd88b29619e1f59` and `cca884d709145f67ac12168038031469c041a9e4` (`folps_v2.0`, 2025-03-05) | `folps/folps.py` plus notebooks/checkpoints | Earlier introductions of `A_full` and the six rows in side histories | No endpoint derivation found |
| `fb093cf22bd429ccc2b1e209d6bfa712300f468f` and `bcf0e38e17f17eddc4731e9e3fa9bc6973cbc4e9` (`update`, 2025-07-24) | `folps/folps.py` and many test/development copies | Reintroduce or copy the same formulas in other side histories | No endpoint derivation found |

Searches for `endpoint`, `symmetri`, `B_sigma`, and the full-A matrix names did
not reveal a symbolic derivation already committed to the repository. The
committed comments/formulas are therefore code-level formulas, not a documented
two-endpoint proof.

## Final Classification

| Channel | Status |
|---|---|
| `I1udd_1` | `REUSE_SCALAR_ROW` |
| `I2uud_1` | `POLARIZE_EXISTING_ROW` |
| `I2uud_2` | `POLARIZE_EXISTING_ROW` |
| `I3uuu_2` | `REUSE_SCALAR_ROW` |
| `I3uuu_3` | `REUSE_SCALAR_ROW` |
| `I1udd_1_b2` | `POLARIZE_EXISTING_ROW` |
| `I2uud_1_b2` | `POLARIZE_EXISTING_ROW` |
| `I2uud_2_b2` | `POLARIZE_EXISTING_ROW` |
| `I1udd_1_bs2` | `POLARIZE_EXISTING_ROW` |
| `I2uud_1_bs2` | `POLARIZE_EXISTING_ROW` |
| `I2uud_2_bs2` | `POLARIZE_EXISTING_ROW` |

No channel requires `EXTRA_CONTRACTION_SAME_MATRIX`, `NEW_ANALYTIC_MATRIX`, or
`UNRESOLVED` under the assumptions in the theory note: equal time, common
velocity field, no velocity bias, plane-parallel LOS, and current FolpsD bias
convention.

## Final Repository Checks

```bash
git diff --stat
```

```text
 .DS_Store      | Bin 10244 -> 12292 bytes
 docs/.DS_Store | Bin 6148 -> 8196 bytes
 2 files changed, 0 insertions(+), 0 deletions(-)
```

```bash
git status --short
```

```text
 M .DS_Store
 M docs/.DS_Store
?? docs/cross_power/
```

No production source file was modified by this Stage 1B audit.
