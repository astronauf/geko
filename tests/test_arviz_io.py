from pathlib import Path

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive

from geko.fitting import _from_numpyro
from geko.postprocess import summarize_posterior


def test_read_legacy_joint_fit():
    path = Path(__file__).resolve().parents[1] / "demo/simple_fit_demo_files/my_galaxy/191250_output_multi"
    with az.from_netcdf(path) as data:
        assert data.posterior.sigma0.shape == (1, 50)
        assert data.prior_predictive.obs_obs1.shape == (1, 1000, 31, 41)
        median = summarize_posterior(data, ["sigma0"])["sigma0"]["50"]
        np.testing.assert_allclose(median, 9.792541015427638)


def test_numpyro_image_prior_round_trip(tmp_path):
    def model():
        offset = numpyro.sample("offset", dist.Normal(0, 1))
        image = numpyro.sample("image", dist.Normal(0, 1).expand((2, 3)))
        numpyro.sample("obs", dist.Normal(offset + image, 1), obs=jnp.zeros((2, 3)))

    prior_key, posterior_key = random.split(random.PRNGKey(42))
    prior = Predictive(model, num_samples=7)(prior_key)
    mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=4, progress_bar=False)
    mcmc.run(posterior_key)

    path = tmp_path / "output"
    _from_numpyro(mcmc, prior).to_netcdf(path)
    with az.from_netcdf(path) as data:
        for name in ("offset", "image"):
            np.testing.assert_array_equal(data.prior[name], prior[name][None, ...])
        np.testing.assert_array_equal(data.prior_predictive.obs, prior["obs"][None, ...])
        assert data.posterior.image.dims[:2] == ("chain", "draw")
        assert data.log_likelihood.obs.shape == (1, 4, 2, 3)
