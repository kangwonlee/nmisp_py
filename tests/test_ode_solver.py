import pathlib
import sys

import numpy as np
import pytest

test_folder_path = pathlib.Path(__file__).parent.resolve()
proj_folder_path = test_folder_path.parent.resolve()
sys.path.append(str(proj_folder_path))


import ode_solver


@pytest.fixture
def t_start() -> float:
    return 0.0


@pytest.fixture
def t_array(t_start:float) -> np.ndarray:
    return np.arange(t_start, 1.05, 0.1)


def test_rk4_with_extra_args(t_start:float, t_array:np.ndarray):

    def f(t, y, a, b):
        return a * t + b * y

    y0 = 1.0

    t, y = ode_solver.rk4(f, t_array=t_array, x_0=y0, args=(2.0, -3.0))
    assert t[0] == t_start
    assert t[-1] == t_array[-1]
    assert len(t) == len(t_array)

    assert y[0] == y0
    assert len(y) == len(t)


def test_rk4_without_extra_args(t_start:float, t_array:np.ndarray):

    a, b = 2.0, -3.0

    def f(t, y):
        return a * t + b * y

    y0 = 1.0

    t, y = ode_solver.rk4(f, t_array=t_array, x_0=y0)
    assert t[0] == t_start
    assert t[-1] == t_array[-1]
    assert len(t) == len(t_array)

    assert y[0] == y0
    assert len(y) == len(t)
