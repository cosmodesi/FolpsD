"""Shared fixtures for the selection tests (NumPy backend; JAX runs in subprocesses)."""
import os
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "numpy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COSMO = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114, f0=0.6880638641959066,
             fnu=0.004453689063655854)
PARS = [1.645, -0.46, -4.0 / 7.0 * (1.645 - 1.0), 32.0 / 315.0 * (1.645 - 1.0),
        3.0, -28.9, 1.5, 0.0, 0.08, -8.1, 1.0 / 0.0002118763, 0.0]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow validation tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("FOLPS_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow validation: use --runslow or FOLPS_RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def linear_input():
    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    return k, pk


@pytest.fixture(scope="session")
def eds_setup(linear_input):
    from folps import MatrixCalculator, NonLinearPowerSpectrumCalculator, build_selection_tables
    k, pk = linear_input
    mm = MatrixCalculator(A_full=True, use_TNS_model=False).get_mmatrices()
    nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **COSMO)
    table, table_now = nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
    tabs = build_selection_tables(nl)
    return dict(mm=mm, nl=nl, table=table, table_now=table_now, tabs=tabs)


@pytest.fixture(scope="session")
def calc():
    from folps import RSDMultipolesPowerSpectrumCalculator
    c = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    c._printed_model_damping_pk = True
    return c


@pytest.fixture(scope="session")
def generic_selection():
    from folps import SelectionBias
    return SelectionBias(s_eta=0.21, s_delta_eta=-0.33, s_eta2=0.27, s_KK_par=0.45, s_Pi2_par=-0.52,
                         s_delta_Pi2_par=0.31, s_eta_Pi2_par=-0.23, s_KPi2_par=0.36, s_Pi3_par=0.19)
