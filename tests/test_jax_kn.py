import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import kn
from scipy.special import kn as scipy_kn


@pytest.mark.parametrize("order", [0, 1])
def test_kn_matches_scipy(order):
    x = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
    np.testing.assert_allclose(kn(order, x), scipy_kn(order, np.asarray(x)), rtol=1e-6)
