r"""Named parameter interfaces and conventions for LOS selection in FolpsD.

Conventions (``docs/selection_1loop_equations.md``):

* Deterministic selection is described by the nine *additional* coordinates of
  :class:`SelectionBias`.  They vanish for ordinary (unselected) redshift-space galaxies.
* They are coordinates in the space of *mapped* DJS coefficients (Desjacques, Jeong &
  Schmidt 2018), in which the standard RSD Jacobian is already included::

      b_eta       = -1   + s_eta
      b_delta_eta = -b1  + s_delta_eta
      b_eta2      =  1   + s_eta2
      b_O         =        s_O          for the six other selection operators

  so that ``s = 0`` gives ordinary redshift-space distortions, *not* real space.
* The ordinary bias parameters (``b1, b2, bK2, btd``) are not duplicated here; they come
  from the legacy FolpsD parameter vector through :func:`folps_bias_to_djs`, the single
  tested adapter.  ``b_delta_eta`` therefore tracks ``b1`` automatically (including under
  differentiation).

All containers are ``NamedTuple``\ s: immutable, JAX pytrees with numerical leaves only.
No code in this package branches on the *values* of these leaves.
"""
from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "SelectionBias", "DJSCoefficients", "RestFrameCoefficients", "EFTExtension",
    "PhysicalDerivativeBias", "SELECTION_FIELDS", "folps_bias_to_djs", "djs_coefficients",
    "rest_frame_coefficients", "quadratic_coefficient_vector", "cubic_coefficient_vector",
    "tree_bA_to_linear_djs", "physical_to_polynomial_derivatives", "selection_Z1",
]


class SelectionBias(NamedTuple):
    """Nine additional deterministic LOS-selection coordinates (all zero = no selection).

    Fields, with the DJS operator each one multiplies:

    ``s_eta``            eta = d_par u_par                 (linear)
    ``s_delta_eta``      delta eta                         (quadratic)
    ``s_eta2``           eta^2                             (quadratic)
    ``s_KK_par``         (K K)_par = n_i K_ik K_kj n_j     (quadratic)
    ``s_Pi2_par``        Pi^[2]_par                        (quadratic)
    ``s_delta_Pi2_par``  delta Pi^[2]_par                  (cubic)
    ``s_eta_Pi2_par``    eta Pi^[2]_par                    (cubic)
    ``s_KPi2_par``       (K Pi^[2])_par                    (cubic)
    ``s_Pi3_par``        Pi^[3]_par                        (cubic)
    """
    s_eta: object = 0.0
    s_delta_eta: object = 0.0
    s_eta2: object = 0.0
    s_KK_par: object = 0.0
    s_Pi2_par: object = 0.0
    s_delta_Pi2_par: object = 0.0
    s_eta_Pi2_par: object = 0.0
    s_KPi2_par: object = 0.0
    s_Pi3_par: object = 0.0

    @classmethod
    def zeros(cls):
        return cls()


SELECTION_FIELDS = SelectionBias._fields


class DJSCoefficients(NamedTuple):
    """Mapped DJS coefficients of the compact one-loop field (equation note E2/E4)."""
    b1: object
    b_eta: object
    b2: object
    b_K2: object
    b_delta_eta: object
    b_eta2: object
    b_KK_par: object
    b_Pi2_par: object
    b_td: object
    b_delta_Pi2_par: object
    b_eta_Pi2_par: object
    b_KPi2_par: object
    b_Pi3_par: object


class RestFrameCoefficients(NamedTuple):
    """Rest-frame (pre-RSD) coefficients of the selected density in the same operator
    basis; galaxies comove with matter.  Related to :class:`DJSCoefficients` by the exact
    second/third-order expansion of the redshift-space map (equation note, bridge section)."""
    b1: object
    bR_eta: object
    b2: object
    b_K2: object
    bR_delta_eta: object
    bR_eta2: object
    bR_KK_par: object
    bR_Pi2_par: object
    b_td: object
    bR_delta_Pi2_par: object
    bR_eta_Pi2_par: object
    bR_KPi2_par: object
    bR_Pi3_par: object


class EFTExtension(NamedTuple):
    """Optional derivative / stochastic terms beyond legacy FolpsD (zero = legacy).

    ``alpha6``          adds ``alpha6 mu^6 k^2 P_L`` to the polynomial counterterms
                        ``(alpha0 + alpha2 mu^2 + alpha4 mu^4) k^2 P_L``  [alpha6] = L^2.
    ``alphashot2_iso``  adds the isotropic ``PshotP * alphashot2_iso * k^2`` noise term
                        (legacy: ``PshotP (alphashot0 + alphashot2 k^2 mu^2)``) [alphashot2_iso] = L^2.
    ``alphashot2_mu4``  adds ``PshotP * alphashot2_mu4 * k^2 mu^4``.  *Proposed*, not part of
                        the DJS stochastic basis: the UV-sensitive order-k^2 part of the selected
                        P22 contains a ``k^2 mu^4`` shape whenever a selection coordinate is
                        nonzero (it vanishes without selection); see the validation report.
    """
    alpha6: object = 0.0
    alphashot2_iso: object = 0.0
    alphashot2_mu4: object = 0.0


class PhysicalDerivativeBias(NamedTuple):
    """DJS higher-derivative coefficients (equation note E9), an *alternative* to the
    polynomial counterterms:

        P_hd = -2 k^2 P_L Z1 [b_nabla2delta - b_eta f mu^2 (beta_nabla2v + beta_par2v mu^2)].

    Using this mode replaces (never adds to) ``alpha0, alpha2, alpha4, alpha6``.
    """
    b_nabla2delta: object = 0.0
    beta_nabla2v: object = 0.0
    beta_par2v: object = 0.0


def folps_bias_to_djs(b1, b2, bs2, b3nl):
    """FolpsD ``(b1, b2, bs2, b3nl)`` -> DJS ``(b1, b2, b_K2, b_td)``.

    Inverse of the adapter used by ``set_bias_scheme(..., 'desi')``:
    ``bs2 = 2 b_K2`` and ``b3nl = -32/21 (b_K2 + 2/5 b_td)``.  Verified against the
    normalization of the actual FolpsD loop templates in
    ``tests/test_selection_zero_selection.py``.
    """
    bK2 = 0.5 * bs2
    btd = -105.0 / 64.0 * b3nl - 2.5 * bK2
    return b1, b2, bK2, btd


def djs_coefficients(b1, b2, bK2, btd, selection):
    """Mapped DJS coefficients from ordinary DJS biases and a :class:`SelectionBias`."""
    s = selection
    return DJSCoefficients(
        b1=b1, b_eta=-1.0 + s.s_eta, b2=b2, b_K2=bK2, b_delta_eta=-b1 + s.s_delta_eta,
        b_eta2=1.0 + s.s_eta2, b_KK_par=s.s_KK_par, b_Pi2_par=s.s_Pi2_par, b_td=btd,
        b_delta_Pi2_par=s.s_delta_Pi2_par, b_eta_Pi2_par=s.s_eta_Pi2_par,
        b_KPi2_par=s.s_KPi2_par, b_Pi3_par=s.s_Pi3_par)


def rest_frame_coefficients(c):
    """Rest-frame coefficients reproducing the mapped field ``c`` through the exact map:
    ``bR_eta = b_eta + 1``, ``bR_delta_eta = b_delta_eta + b1``,
    ``bR_eta2 = b_eta2 + b_eta`` (``= b_eta2 - 1 + bR_eta``), ``bR_eta_Pi2 = b_eta_Pi2 + b_Pi2``,
    all others identical."""
    return RestFrameCoefficients(
        b1=c.b1, bR_eta=c.b_eta + 1.0, b2=c.b2, b_K2=c.b_K2, bR_delta_eta=c.b_delta_eta + c.b1,
        bR_eta2=c.b_eta2 + c.b_eta, bR_KK_par=c.b_KK_par, bR_Pi2_par=c.b_Pi2_par, b_td=c.b_td,
        bR_delta_Pi2_par=c.b_delta_Pi2_par, bR_eta_Pi2_par=c.b_eta_Pi2_par + c.b_Pi2_par,
        bR_KPi2_par=c.b_KPi2_par, bR_Pi3_par=c.b_Pi3_par)


def quadratic_coefficient_vector(c, galilean_safe=True):
    """Coefficients of the ten quadratic kernels.

    ``galilean_safe=False``: the equation-note basis (E5),
    ``c_2 = (b1, b_eta, b2/2, b_K2, b_delta_eta, b_eta2, b_KK, b_Pi2, -b1, -b_eta)``.
    ``galilean_safe=True`` (stored tensor basis): the advection kernels are combined with
    ``delta eta`` and ``eta^2`` into kernels suppressed at large ``q``, giving
    ``(b1, b_eta, b2/2, b_K2, b_delta_eta + b1, b_eta2 + b_eta, b_KK, b_Pi2, -b1, -b_eta)``.
    """
    if galilean_safe:
        return (c.b1, c.b_eta, 0.5 * c.b2, c.b_K2, c.b_delta_eta + c.b1, c.b_eta2 + c.b_eta, c.b_KK_par,
                c.b_Pi2_par, -c.b1, -c.b_eta)
    return (c.b1, c.b_eta, 0.5 * c.b2, c.b_K2, c.b_delta_eta, c.b_eta2, c.b_KK_par, c.b_Pi2_par,
            -c.b1, -c.b_eta)


def cubic_coefficient_vector(c):
    """``c_13`` multiplying the sixteen reduced cubic operators (equation note E7)."""
    return (c.b1, c.b_eta, c.b_K2, c.b_delta_eta, c.b_eta2, c.b_KK_par, c.b_td, c.b_delta_Pi2_par,
            c.b_eta_Pi2_par, c.b_KPi2_par, -c.b_Pi2_par, -c.b1, -c.b_eta, -c.b_eta, -c.b_Pi2_par,
            c.b_Pi3_par + 2.0 * c.b_Pi2_par)


def selection_Z1(c, f, mu):
    """Linear kernel ``Z1 = b1 - b_eta f mu^2``."""
    return c.b1 - c.b_eta * f * mu ** 2


def tree_bA_to_linear_djs(b1_folps, bA, f):
    """Historical tree-only traceless parameter ``bA`` -> matched linear DJS coordinates.

    ``b1F + f mu^2 + bA (mu^2 - 1/3) = b1D - b_eta f mu^2`` with
    ``b1D = b1F - bA/3`` and ``b_eta = -1 - bA/f``, i.e. ``s_eta = -bA / f``.

    This is a *linear*, constant-``f`` identity only.  It does not determine any nonlinear
    selection coefficient and must not be used with a scale-dependent ``f``.
    Returns ``(b1_djs, s_eta)``.
    """
    return b1_folps - bA / 3.0, -bA / f


def physical_to_polynomial_derivatives(b1, b_eta, f, phys):
    """Convert :class:`PhysicalDerivativeBias` to polynomial coefficients ``(c0, c2, c4, c6)``
    of ``k^2 P_L mu^{2r}`` (equation note E9), with ``a = b_eta f``::

        (c0, c2, c4, c6) = (-2 b1 d, 2a (d + b1 v), 2 b1 a w - 2 a^2 v, -2 a^2 w).
    """
    a = b_eta * f
    d, v, w = phys.b_nabla2delta, phys.beta_nabla2v, phys.beta_par2v
    return (-2.0 * b1 * d, 2.0 * a * (d + b1 * v), 2.0 * b1 * a * w - 2.0 * a * a * v, -2.0 * a * a * w)
