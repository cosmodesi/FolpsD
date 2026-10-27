r"""Full EdS one-loop LOS-selection model on top of the FolpsD EFT power spectrum.

The selected spectrum is the ordinary FolpsD EFT prediction evaluated at the *same*
ordinary biases, plus selected-minus-unselected increments computed in one matched
convention and mixed with the existing IR prescription (equation note E8/E11):

    P_IR(sel) = P_IR^FolpsD(b0)
              + (Z1^2 - Z1,0^2) {P_nw + E (1 + A) P_w}                          tree increment
              + Delta L[P_nw] + E (Delta L[P_L] - Delta L[P_nw])                 loop increment
              + [ctilde, alpha6, physical-derivative, isotropic-noise extensions]

with ``Delta L = L(b) - L(b0)`` from :mod:`folps.selection_templates`.  The BAO
displacement variance ``Sigma_s^2(mu)`` is the physical one: selection does not enter it.
GTNS is part of the complete loop of FolpsD's EFT branch; the increment carries the
selection response of the complete loop, so no extra GTNS-like term is added.

Supported: ``model="EFT"`` auto spectra, EdS tables (``kernels='EdS'``), ``A_full=True``,
``use_TNS_model=False``, NumPy and JAX backends.  Everything else raises.
"""
from __future__ import annotations

import numpy as _np

from .selection_operators import (EFTExtension, PhysicalDerivativeBias, SelectionBias,
                                  djs_coefficients, folps_bias_to_djs,
                                  physical_to_polynomial_derivatives)
from .selection_templates import (SelectionLoopTables, interp_selection_table,
                                  selection_loop, selection_loop_increment)

__all__ = ["selected_rsd_pkmu", "validate_selection_request"]


def _is_concrete(x):
    try:
        _np.asarray(x, dtype=float)
        return True
    except Exception:
        return False


def validate_selection_request(calc, *, selection, selection_tables, bA, pars_b, damping_method,
                               eft_extension, physical_derivatives, use_GTNS=None):
    """Static validation of a selection-mode request (never inspects traced values)."""
    from . import folps as _f
    if selection is not None and not isinstance(selection, SelectionBias):
        raise TypeError("selection must be a folps.SelectionBias (use SelectionBias() for an explicit zero)")
    if eft_extension is not None and not isinstance(eft_extension, EFTExtension):
        raise TypeError("eft_extension must be a folps.EFTExtension")
    if physical_derivatives is not None and not isinstance(physical_derivatives, PhysicalDerivativeBias):
        raise TypeError("physical_derivatives must be a folps.PhysicalDerivativeBias")
    if calc.model != "EFT":
        raise NotImplementedError(
            f"The one-loop selection model is implemented for model='EFT' only (no FoG damping); "
            f"got model={calc.model!r}.")
    if pars_b is not None:
        raise NotImplementedError(
            "Selected cross spectra are not implemented; the auto-spectrum selection formula must not "
            "be reused for cross spectra. Call without selection/eft_extension for legacy cross spectra.")
    if bA is not None:
        raise ValueError(
            "bA (historical tree-only anisotropic bias) cannot be combined with selection/eft_extension/"
            "physical_derivatives. Use folps.tree_bA_to_linear_djs to obtain the matched linear coordinates.")
    if not getattr(_f, "A_full_status", True):
        raise ValueError("selection mode requires MatrixCalculator(A_full=True): the selected loop increment is "
                         "added to the complete perturbative baseline")
    if getattr(_f, "use_TNS_model_status", False):
        raise ValueError("selection mode requires MatrixCalculator(use_TNS_model=False) (complete one-loop baseline)")
    if selection is not None and use_GTNS is False:
        # The selected-minus-unselected loop increment is built from the complete exact-map loop, which contains
        # the selection response of the perturbative velocity-dispersion (GTNS) term; removing GTNS from the
        # baseline alone would be inconsistent, and the selection-dependent GTNS subtraction is not implemented.
        raise NotImplementedError(
            "use_GTNS=False is not supported with selection: the selected one-loop increment includes the "
            "selection response of GTNS. Use use_GTNS=None (default; GTNS kept, model='EFT') or True.")
    if selection is not None:
        if not isinstance(selection_tables, SelectionLoopTables):
            raise ValueError("selection requires selection_tables=folps.build_selection_tables(nonlinear)")
        for t in selection_tables:
            if t.metadata.get("kernels") != "EdS":
                raise NotImplementedError("selection tables must use EdS kernels")


def _same_array(a, b, rtol=1e-12):
    """True / False for concrete arrays; None when either is traced (cannot be compared by value)."""
    if a is b:
        return True
    if not (_is_concrete(a) and _is_concrete(b)):
        return None
    a, b = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(_np.allclose(a, b, rtol=rtol, atol=0.0))


def _check_tables_compatible(table, selection_tables, table_now=None):
    """Pairing checks between the legacy (baseline) tables and the selection tables.

    The selected spectrum is ``P_legacy[table] + (P_selected - P_unselected)[selection_tables]``,
    so both must describe the same linear input.  On concrete arrays this compares directly the
    wavenumber grid, the linear spectrum of the full table, the grid and spectrum of the no-wiggle
    table, and ``f0`` (the selection tables store the same arrays as the raw legacy tables when
    built from the same ``calculate_loop_table`` call, so agreement is to round-off).  Under JAX
    tracing values cannot be compared: only shapes are checked, and pairing is the caller's
    responsibility (build both from the same calculator call, as the desilike wrapper does).
    """
    fk_col = table[2]  # raw legacy table: (k, P_L, f(k)/f0, ...)
    if _is_concrete(fk_col):
        if not _np.allclose(_np.asarray(fk_col, dtype=float), 1.0, rtol=0, atol=1e-12):
            raise NotImplementedError(
                "The legacy table has scale-dependent growth (kernels='fk'); full one-loop selection is only "
                "derived for EdS kernels. Build the legacy table with kernels='EdS'.")
    f0 = table[-1]
    fs = selection_tables.full.f0
    if _is_concrete(f0) and _is_concrete(fs):
        if not _np.isclose(float(f0), float(fs), rtol=1e-12, atol=0):
            raise ValueError(f"growth rate mismatch between legacy table (f0={float(f0)}) and selection tables "
                             f"(f0={float(fs)}); build both from the same calculator call")
    pairs = [("k grid of the full table", table[0], selection_tables.full.k),
             ("linear spectrum of the full table", table[1], selection_tables.full.pk)]
    if table_now is not None:
        pairs += [("k grid of the no-wiggle table", table_now[0], selection_tables.nowiggle.k),
                  ("no-wiggle linear spectrum", table_now[1], selection_tables.nowiggle.pk)]
    for name, a, b in pairs:
        if _np.shape(a) != _np.shape(b):
            raise ValueError(f"legacy and selection tables do not match ({name}: shapes {_np.shape(a)} vs "
                             f"{_np.shape(b)}); build both from the same calculator call")
        if _same_array(a, b) is False:
            raise ValueError(f"legacy and selection tables do not match ({name} differs): the selected increment "
                             "would be added to a baseline for another linear input. Build table/table_now and "
                             "selection_tables from the same NonLinearPowerSpectrumCalculator.calculate_loop_table call")


def selected_rsd_pkmu(calc, k, mu, pars, table, table_now, IR_resummation=True, use_GTNS=None, *,
                      selection=None, selection_tables=None, eft_extension=None, physical_derivatives=None,
                      return_components=False):
    """Selected redshift-space ``P(k, mu)`` (see module docstring).  ``pars`` are FolpsD-basis
    parameters (after ``set_bias_scheme``)."""
    from . import folps as _f
    np = _f.np
    params = calc._split_power_pars(pars)
    b1, b2, bs2, b3nl = params.bias
    (alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p) = params.nuisance
    ext = eft_extension if eft_extension is not None else EFTExtension()

    tab = calc.interp_table(k, table, _f.A_full_status)
    tab_now = calc.interp_table(k, table_now, _f.A_full_status)
    f0 = tab[-1]
    pkl, pkl_now = tab[0], tab_now[0]
    sigma2, delta_sigma2 = tab_now[-3:-1]
    sigma2w, sigma2w_now = tab[-2], tab_now[-4]
    if IR_resummation:
        sigma2t = (1 + f0 * mu ** 2 * (2 + f0)) * sigma2 + (f0 * mu) ** 2 * (mu ** 2 - 1) * delta_sigma2
    else:
        sigma2t = 0.0 * mu
    E = np.exp(-k ** 2 * sigma2t)
    tree_ir = pkl_now + E * (pkl - pkl_now) * (1 + k ** 2 * sigma2t)

    b1d, b2d, bK2, btd = folps_bias_to_djs(b1, b2, bs2, b3nl)
    sel = selection if selection is not None else SelectionBias()
    c = djs_coefficients(b1d, b2d, bK2, btd, sel)
    c0 = djs_coefficients(b1d, b2d, bK2, btd, SelectionBias())

    # ---------------------------------------------------------------- derivative sector
    alpha6 = ext.alpha6
    pars_eff = list(pars)
    if physical_derivatives is not None:
        for name, val in (("alpha0", alpha0), ("alpha2", alpha2), ("alpha4", alpha4), ("alpha6", ext.alpha6)):
            # Checked on concrete values only; a traced value cannot be checked and would be silently replaced,
            # so it is rejected: in this mode the polynomial counterterms are not parameters at all.
            if not _is_concrete(val):
                raise TypeError(f"physical_derivatives replaces the polynomial counterterms, so {name} must be a "
                                "concrete 0 (not a traced/sampled value): do not include alpha0, alpha2, alpha4 or "
                                "alpha6 among the differentiated or jitted arguments in this mode.")
            if float(val) != 0.0:
                raise ValueError(f"physical_derivatives replaces the polynomial counterterms; {name} must be 0 "
                                 f"(got {float(val)}). Never sum both parameterizations of the derivative sector.")
        cpoly = physical_to_polynomial_derivatives(c.b1, c.b_eta, f0, physical_derivatives)
        pars_eff[4], pars_eff[5], pars_eff[6] = cpoly[0], cpoly[1], cpoly[2]
        alpha6 = cpoly[3]

    base = calc.get_rsd_pkmu(k, mu, pars_eff, table, table_now, IR_resummation, damping=None,
                             damping_method=None, use_GTNS=use_GTNS)
    ct6 = alpha6 * mu ** 6 * k ** 2 * (E * pkl + (1 - E) * pkl_now)
    noise_ext = PshotP * (ext.alphashot2_iso + ext.alphashot2_mu4 * mu ** 4) * k ** 2
    out = base + ct6 + noise_ext

    if selection is not None:
        _check_tables_compatible(table, selection_tables, table_now)
        Z1 = c.b1 - c.b_eta * f0 * mu ** 2
        Z10 = c0.b1 - c0.b_eta * f0 * mu ** 2
        dZ2 = (Z1 - Z10) * (Z1 + Z10)
        d_tree = dZ2 * tree_ir
        ip_full = interp_selection_table(k, selection_tables.full)
        ip_now = interp_selection_table(k, selection_tables.nowiggle)
        d22_w, d13_w = selection_loop_increment(k, mu, selection_tables.full, c, c0, ip_full)
        d22_n, d13_n = selection_loop_increment(k, mu, selection_tables.nowiggle, c, c0, ip_now)
        d_loop = (d22_n + d13_n) + E * ((d22_w + d13_w) - (d22_n + d13_n))
        d_ctilde = ctilde * (mu * k * f0) ** 4 * dZ2 * (E * sigma2w ** 2 * pkl + (1 - E) * sigma2w_now ** 2 * pkl_now)
        out = out + d_tree + d_loop + d_ctilde
    if not return_components:
        return out

    # ---------------------------------------------------------------- component output
    zero_lin = list(pars_eff)
    zero_lin[4] = zero_lin[5] = zero_lin[6] = 0.0
    zero_lin[7] = 0.0
    zero_lin[8] = zero_lin[9] = 0.0
    # same options as the legacy EFT path of get_rsd_pkmu (damping_method forced to 'loop+ctr', use_GTNS forwarded)
    zl = calc.get_eft_pkmu(k, mu, zero_lin, tab, damping=None, damping_method='loop+ctr', use_GTNS=use_GTNS)
    zl_now = calc.get_eft_pkmu(k, mu, zero_lin, tab_now, damping=None, damping_method='loop+ctr', use_GTNS=use_GTNS)
    loop_base = E * zl + (1 - E) * zl_now
    Z10 = c0.b1 - c0.b_eta * f0 * mu ** 2
    comp = {"tree_baseline": Z10 ** 2 * tree_ir, "loop_baseline": loop_base}
    a0, a2, a4 = pars_eff[4], pars_eff[5], pars_eff[6]
    comp["counterterms"] = (a0 + a2 * mu ** 2 + a4 * mu ** 4 + alpha6 * mu ** 6) * k ** 2 * (E * pkl + (1 - E) * pkl_now)
    comp["stochastic"] = PshotP * (alphashot0 + alphashot2 * (k * mu) ** 2
                                   + (ext.alphashot2_iso + ext.alphashot2_mu4 * mu ** 4) * k ** 2)
    comp["ctilde"] = ctilde * (mu * k * f0) ** 4 * Z10 ** 2 * (E * sigma2w ** 2 * pkl + (1 - E) * sigma2w_now ** 2 * pkl_now)
    if selection is not None:
        comp["delta_tree"] = d_tree
        comp["delta_loop"] = d_loop
        comp["delta_ctilde"] = d_ctilde
        Z1 = c.b1 - c.b_eta * f0 * mu ** 2
        comp["tree"] = Z1 ** 2 * tree_ir
        p22w, p13w = selection_loop(k, mu, selection_tables.full, c, ip_full)
        p22n, p13n = selection_loop(k, mu, selection_tables.nowiggle, c, ip_now)
        comp["P22_templates"] = p22n + E * (p22w - p22n)
        comp["P13_templates"] = p13n + E * (p13w - p13n)
        comp["loop"] = loop_base + d_loop
    else:
        comp["tree"] = comp["tree_baseline"]
        comp["loop"] = loop_base
    comp["total"] = out
    comp["scheme"] = "P22: zero-lag constant subtracted; P13: bare k^2 sigma^2 kept, sigma_0^2 constants renormalized"
    return comp


LINEAR_PARAMETERS = ("alpha0", "alpha2", "alpha4", "alphashot0", "alphashot2")
EXTENDED_LINEAR_PARAMETERS = ("alpha6", "alphashot2_iso", "alphashot2_mu4")
_LEGACY_INDEX = {"alpha0": 4, "alpha2": 5, "alpha4": 6, "alphashot0": 8, "alphashot2": 9}


def selected_pkell_linear_decomposition(calc, kobs, qpar, qper, pars, table, table_now, *, bias_scheme="folps",
                                        nmu=6, ells=(0, 2, 4), IR_resummation=True, selection=None,
                                        selection_tables=None, eft_extension=None, include_extended=None):
    """Exact decomposition ``P_ell = P_ell,const + sum_i theta_i T_ell,i`` of the (selected)
    multipoles in the genuinely linear EFT/stochastic parameters.

    The templates are differences of full-model evaluations, so they include AP, the IR
    mixing and the selection exactly (the model is linear in these parameters).  The
    nonlinear parameters (biases, selection coordinates, ``ctilde``) are held fixed.
    Returns ``(P_const, templates, names)`` with ``templates`` of shape
    ``(n_param, len(ells), len(kobs))``.
    """
    np = __import__("folps.folps", fromlist=["np"]).np
    pars = list(calc.set_bias_scheme(list(pars), bias_scheme=bias_scheme))
    if include_extended is None:
        include_extended = eft_extension is not None
    names = list(LINEAR_PARAMETERS) + (list(EXTENDED_LINEAR_PARAMETERS) if include_extended else [])
    base = list(pars)
    for n in LINEAR_PARAMETERS:
        base[_LEGACY_INDEX[n]] = 0.0
    ext0 = EFTExtension() if (include_extended or eft_extension is not None) else None

    def run(p, ext):
        return calc.get_rsd_pkell(kobs, qpar, qper, p, table, table_now, bias_scheme="folps", damping=None,
                                  nmu=nmu, ells=ells, IR_resummation=IR_resummation, selection=selection,
                                  selection_tables=selection_tables, eft_extension=ext)
    pconst = run(base, ext0)
    temps = []
    for n in names:
        p, ext = list(base), ext0
        if n in _LEGACY_INDEX:
            p[_LEGACY_INDEX[n]] = 1.0
        else:
            ext = ext0._replace(**{n: 1.0})
        temps.append(run(p, ext) - pconst)
    return pconst, np.stack(temps), names
