#!/usr/bin/env python
"""Compare the generated P22 tensor with the DJS supplement's ``Anmp`` coefficients.

Reads the Mathematica notebook ``pkgs_supplement.nb`` of
https://github.com/djeong98/pkgs_supplement (tested at commit 2ecd100b2ad68ec7e1e19c6d56aa45bd01dcf682,
sha256 638af6d8afa2367d10c625e1394a3ad302f1b8806ab0db528b8f2b12149e3f14) as *data*: the box
expression of the ``Anmp = {...}`` cell is parsed into SymPy polynomials in the DJS biases
and f.  Nothing from the notebook is executed.

Usage::

    python dev/selection_audit/compare_djs_supplement.py /path/to/pkgs_supplement.nb [--out result.json]
"""
import argparse
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[2]

# DJS symbol -> our DJSCoefficients name
SYMBOLS = {"1": "b1", "eta": "b_eta", "2": "b2", "K2": "b_K2", "Pi2p": "b_Pi2_par", "K2p": "b_KK_par",
           "deltaeta": "b_delta_eta", "eta2": "b_eta2"}


def tokenize(s):
    toks = re.findall(r'"(?:[^"\\]|\\.)*"|[A-Za-z][A-Za-z0-9]*|[\[\]{},]', s)
    return toks


def parse(tokens, i=0):
    t = tokens[i]
    if t.startswith('"'):
        return ("str", t[1:-1].replace('\\"', '"')), i + 1
    if t == "{":
        items, i = [], i + 1
        while tokens[i] != "}":
            e, i = parse(tokens, i)
            items.append(e)
            if tokens[i] == ",":
                i += 1
        return ("list", items), i + 1
    name = t
    assert tokens[i + 1] == "[", (name, tokens[i:i + 5])
    args, i = [], i + 2
    while tokens[i] != "]":
        e, i = parse(tokens, i)
        args.append(e)
        if tokens[i] == ",":
            i += 1
    return (name, args), i + 1


def text(node):
    kind = node[0]
    if kind == "str":
        return node[1]
    if kind == "list":
        return "".join(text(x) for x in node[1])
    return "".join(text(x) for x in node[1])


def label(node):
    t = text(node)
    t = t.replace("\\[Delta]", "delta").replace("\\[Eta]", "eta").replace("\\[CapitalPi]", "Pi")
    return re.sub(r"[^A-Za-z0-9]", "", t)


def to_py(node):
    kind = node[0]
    if kind == "str":
        s = node[1]
        return {"{": "[", "}": "]", " ": "*", "\\[IndentingNewLine]": ""}.get(s, s)
    if kind == "list":
        return "[" + ",".join(to_py(x) for x in node[1]) + "]"
    name, args = node
    if name == "RowBox":
        parts = [to_py(x) for x in args[0][1]]
        out = ""
        for p in parts:
            if out and p and out[-1] not in "+-*/(,[ " and p[0] not in "+-*/),]":
                out += "*"
            out += p
        raw = [x for x in args[0][1]]
        has_comma = any(x[0] == "str" and x[1] == "," for x in raw)
        has_bracket = any(x[0] == "str" and x[1] in ("{", "}") for x in raw)
        return out if (has_comma or has_bracket) else "(" + out + ")"
    if name == "FractionBox":
        return f"(({to_py(args[0])})/({to_py(args[1])}))"
    if name == "SuperscriptBox":
        return f"(({to_py(args[0])})**({to_py(args[1])}))"
    if name == "SubscriptBox":
        return f"S_{label(args[1])}"
    if name == "SubsuperscriptBox":
        return f"((S_{label(args[1])})**({to_py(args[2])}))"
    if name in ("FormBox", "TagBox", "StyleBox", "InterpretationBox"):
        return to_py(args[0])
    raise ValueError(f"unsupported box {name}")


def load_anmp(nb_path):
    s = Path(nb_path).read_text()
    start = s.index('RowBox[{"Anmp", "=", ')
    toks = tokenize(s[start:start + 3_000_000])
    tree, _ = parse(toks, 0)
    rhs = tree[1][0][1][2]                      # RowBox[{ "Anmp", "=", <rhs> }]
    py = to_py(rhs).replace("[", "[").replace("]", "]")
    syms = {f"S_{k}": sp.Symbol(v) for k, v in SYMBOLS.items()}
    syms["f"] = sp.Symbol("f")
    return eval(py, {"__builtins__": {}}, syms)


def our_weights():
    """W[n][(m, r)] as SymPy polynomials, from the generated rational tensor."""
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location("data", ROOT / "folps" / "_selection_coefficients_data.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    b = {n: sp.Symbol(n) for n in SYMBOLS.values()}
    f = sp.Symbol("f")
    c = [b["b1"], b["b_eta"], b["b2"] / 2, b["b_K2"], b["b_delta_eta"] + b["b1"], b["b_eta2"] + b["b_eta"],
         b["b_KK_par"], b["b_Pi2_par"], -b["b1"], -b["b_eta"]]
    W = {}
    for (a, bb, n, m, r), (num, den) in d.P22_TENSOR.items():
        mult = 1 if a == bb else 2
        W.setdefault((n, m, r), 0)
        W[(n, m, r)] += mult * sp.Rational(num, den) * c[a] * c[bb] * f ** (d.QUADRATIC_NF[a] + d.QUADRATIC_NF[bb])
    return {k: sp.expand(v) for k, v in W.items()}


def weights_with_signs(signs):
    """Our P22 weights in the equation-note kernel basis with optional sign flips s_a of the
    individual kernels S_a (a = 0..9 in the E5 order)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("data", ROOT / "folps" / "_selection_coefficients_data.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    b = {n: sp.Symbol(n) for n in SYMBOLS.values()}
    f = sp.Symbol("f")
    cp = [b["b1"], b["b_eta"], b["b2"] / 2, b["b_K2"], b["b_delta_eta"], b["b_eta2"], b["b_KK_par"],
          b["b_Pi2_par"], -b["b1"], -b["b_eta"]]
    cp = [sg * x for sg, x in zip(signs, cp)]
    cg = list(cp)                       # stored tensor uses S'_8 = S_8 + S_4, S'_9 = S_9 + S_5
    cg[4], cg[5] = cp[4] - cp[8], cp[5] - cp[9]
    W = {}
    for (a, bb, n, m, r), (num, den) in d.P22_TENSOR.items():
        mult = 1 if a == bb else 2
        W[(n, m, r)] = W.get((n, m, r), 0) + mult * sp.Rational(num, den) * cg[a] * cg[bb] * f ** (
            d.QUADRATIC_NF[a] + d.QUADRATIC_NF[bb])
    return W


def main():
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sha = hashlib.sha256(Path(args.notebook).read_bytes()).hexdigest()
    A = load_anmp(args.notebook)
    dims = (len(A), len(A[0]), len(A[0][0]))
    rnd = random.Random(1)
    names = list(SYMBOLS.values()) + ["f"]
    vals = [{sp.Symbol(n): rnd.uniform(-1.5, 1.5) for n in names} for _ in range(3)]
    theirs = [[[[float(sp.sympify(A[n][m][p]).subs(v)) for p in range(dims[2])] for m in range(dims[1])]
               for n in range(dims[0])] for v in vals]
    kernels = ["F2", "eta^(2)", "delta^2", "K^2", "delta eta", "eta^2", "(KK)_par", "Pi2_par",
               "u_par d_par delta", "u_par d_par eta"]
    hyps = {"identity": [1] * 10}
    for a in range(10):
        sg = [1] * 10
        sg[a] = -1
        hyps["flip " + kernels[a]] = sg
    result = {}
    for name, sg in hyps.items():
        W = weights_with_signs(sg)
        worst = 0.0
        for t, v in zip(theirs, vals):
            for n in range(dims[0]):
                for m in range(dims[1]):
                    for p in range(dims[2]):
                        mine = float(sp.sympify(W.get((n, m, p), 0)).subs(v))
                        worst = max(worst, abs(t[n][m][p] - mine) / max(1.0, abs(mine)))
        result[name] = worst
        print(f"{name:28s} worst relative mismatch over all {dims[0]*dims[1]*dims[2]} entries: {worst:.2e}", flush=True)
    report = {"notebook_sha256": sha, "Anmp_dims": dims, "index_convention": "Anmp[[n+1, m+1, r+1]] = mu^2n nu^m I_mr",
              "hypotheses_worst_mismatch": result}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()


def numeric_compare(nb_path, ntrial=5, seed=0):
    """Compare entry by entry at random numerical bias values, for f and for -f in the
    supplement (velocity-sign convention test).  Returns the worst relative mismatch for
    each hypothesis."""
    import random
    A = load_anmp(nb_path)
    ours = our_weights()
    names = list(SYMBOLS.values()) + ["f"]
    rnd = random.Random(seed)
    worst = {"f": 0.0, "-f": 0.0}
    for _ in range(ntrial):
        vals = {sp.Symbol(n): rnd.uniform(-1.5, 1.5) for n in names}
        vals_neg = dict(vals)
        vals_neg[sp.Symbol("f")] = -vals[sp.Symbol("f")]
        for n in range(len(A)):
            for m in range(len(A[0])):
                for p in range(len(A[0][0])):
                    t = sp.sympify(A[n][m][p])
                    mine = float(sp.sympify(ours.get((n, m, p), 0)).subs(vals))
                    scale = max(abs(mine), 1e-12)
                    for tag, v in (("f", vals), ("-f", vals_neg)):
                        th = float(t.subs(v))
                        worst[tag] = max(worst[tag], abs(th - mine) / scale if abs(mine) > 1e-12 else abs(th))
    return worst
