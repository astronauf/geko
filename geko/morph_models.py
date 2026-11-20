"""
Morphology models for geko.

Adding a new model: subclass MorphologyModel, define _DEFAULT_PARAMETERS,
sample(), and generate_flux_map(), then add to MORPH_REGISTRY.
"""

__all__ = ['MorphologyModel', 'SersicMorphology', 'MORPH_REGISTRY']

from abc import abstractmethod
import jax.numpy as jnp

from .param_spec import BaseModel, ParameterSpec
from . import utils


class MorphologyModel(BaseModel):
    """Abstract base for a morphological model.

    sample() returns a dict of morph_params.
    generate_flux_map() returns the oversampled 2D flux array.
    """

    @abstractmethod
    def sample(self) -> dict:
        ...

    @abstractmethod
    def generate_flux_map(self, morph_params: dict, shared_params: dict,
                          image_shape: int, factor: int) -> jnp.ndarray:
        ...

    @abstractmethod
    def adjust_for_observation(self, morph_params: dict, theta_rot_deg: float,
                               center: float) -> dict:
        """Return a copy of morph_params adjusted for a specific observation orientation.

        Called once per observation in the multi-obs inference loop so that
        spatially-dependent parameters (position angle, centroids) are rotated
        into the frame of each grism observation.

        Parameters
        ----------
        morph_params : dict
            Parameters as sampled in the reference frame.
        theta_rot_deg : float
            Rotation angle of this observation relative to the reference frame (degrees).
        center : float
            Image centre coordinate (pixels), used as rotation pivot.
        """
        ...

    def set_priors_from_pysersic(self, catalog_row: dict):
        """Apply morphological priors from a pysersic-style catalog dict.

        Keys should follow the pattern {param_name}_{mu|std|min|max},
        e.g. 'r_eff_mu', 'r_eff_std', 'n_mu', 'PA_morph_mu', etc.
        Unknown keys are silently ignored.
        """
        self.apply_prior_overrides(catalog_row)


class SersicMorphology(MorphologyModel):
    """Parametric Sersic morphology model.

    Rendered as a pixel-integrated mixture of Gaussians (utils.sersic_image), which
    is accurate for n >= utils.SERSIC_N_MIN (0.65, as in PySersic); the n prior is
    never allowed below that.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enforce_n_min()

    def apply_prior_overrides(self, overrides: dict):
        super().apply_prior_overrides(overrides)
        self._enforce_n_min()

    def _enforce_n_min(self):
        for spec in self.parameters:
            if spec.name == 'n' and (spec.prior_min is None or spec.prior_min < utils.SERSIC_N_MIN):
                if spec.prior_min is not None:
                    import warnings
                    warnings.warn(f"Sersic n prior minimum {spec.prior_min} is below {utils.SERSIC_N_MIN}, "
                                  f"where the Gaussian-mixture renderer is accurate; raising it to {utils.SERSIC_N_MIN}.")
                spec.prior_min = utils.SERSIC_N_MIN

    _DEFAULT_PARAMETERS = [
        ParameterSpec('amplitude', r'$A$',                    r'$A$',
                      'TruncatedNormal'),
        ParameterSpec('r_eff',    r'$r_{\rm e}$ [px]',        r'$r_{\rm e}$',
                      'TruncatedNormal'),
        ParameterSpec('n',        r'$n$',                     r'$n$',
                      'TruncatedNormal'),
        ParameterSpec('PA_morph', r'PA$_{\rm morph}$ [deg]',  r'PA$_{\rm morph}$',
                      'Normal'),
        ParameterSpec('xc_morph', r'$x_0$ [px]',              r'$x_0$',
                      'Normal'),
        ParameterSpec('yc_morph', r'$y_0$ [px]',              r'$y_0$',
                      'Normal'),
    ]

    def adjust_for_observation(self, morph_params: dict, theta_rot_deg: float,
                               center: float) -> dict:
        """Rotate PA_morph and centroids into this observation's frame."""
        import jax.numpy as jnp
        theta_rad = jnp.radians(theta_rot_deg)
        obs_params = dict(morph_params)
        # PA is measured from +y towards +x, so the clockwise rotate_coords(theta)
        # applied to the centroids below turns it by +theta
        obs_params['PA_morph'] = morph_params['PA_morph'] + theta_rot_deg
        xc_obs, yc_obs = utils.rotate_coords(
            morph_params['xc_morph'], morph_params['yc_morph'],
            center, center, theta_rad,
        )
        obs_params['xc_morph'] = xc_obs
        obs_params['yc_morph'] = yc_obs
        return obs_params

    def sample(self) -> dict:
        return self._sample()

    def sample_without_amplitude(self) -> dict:
        """Sample all parameters except amplitude (used in multi-obs fitting)."""
        from .param_spec import sample_specs
        specs = [s for s in self.parameters if s.name != 'amplitude']
        return sample_specs(specs)

    def generate_flux_map(self, morph_params: dict, shared_params: dict,
                          image_shape: int, factor: int) -> jnp.ndarray:
        amplitude = morph_params['amplitude']
        r_eff     = morph_params['r_eff']
        n         = morph_params['n']
        PA_morph  = morph_params['PA_morph']
        xc_morph  = morph_params['xc_morph']
        yc_morph  = morph_params['yc_morph']
        i         = shared_params['i']

        ellip = 1.0 - utils.compute_axis_ratio(inc=i, q0=0.2)
        # pixel-integrated Gaussian-mixture Sersic on the subpixel-centred model grid
        return utils.sersic_image(image_shape, factor, amplitude, r_eff, n,
                                  xc_morph, yc_morph, ellip,
                                  (90.0 - PA_morph) * jnp.pi / 180.0)


class EdgeDisk(MorphologyModel):
    """
    Parametric edge-on sech^2 disk morphology model.
    van der Kruit & Searle (1981) showed that an edge-on (i=90 radially exponential (n=1) disk can be described by

    I(R, h) = I_0 ⋅ R / Rd ⋅ K_1(R / R_d) ⋅ sech^2(h / h_0)

    where I_0 is the central density,
    K_1 is the modified Bessel function of the second kind of order one,
    R_d is the exponential scale length in the disk plane, and
    h_0 is the scale height

    """
    _DEFAULT_PARAMETERS = [
        ParameterSpec("r_scale", r"$R_d$", r"$R_d$", "Uniform"),
        ParameterSpec("h_scale", r"$h_0$", r"$h_0$", "Uniform"),
        ParameterSpec("amplitude", r"$A$", r"$A$", "TruncatedNormal"),
        ParameterSpec('xc_morph', r'$x_0$ [px]', r'$x_0$', 'Normal'),
        ParameterSpec('yc_morph', r'$y_0$ [px]', r'$y_0$', 'Normal'),
        ParameterSpec('PA_morph', r'PA$_{\rm morph}$ [deg]', r'PA$_{\rm morph}$', 'Normal'),
    ]

    def adjust_for_observation(self, morph_params: dict, theta_rot_deg: float, center: float) -> dict:
        """Rotate PA_morph and centroids into this observation's frame."""
        theta_rad = jnp.radians(theta_rot_deg)
        obs_params = dict(morph_params)
        obs_params['PA_morph'] = morph_params['PA_morph'] - theta_rot_deg
        xc_obs, yc_obs = utils.rotate_coords(
            morph_params['xc_morph'], morph_params['yc_morph'],
            center, center, theta_rad,
        )
        obs_params['xc_morph'] = xc_obs
        obs_params['yc_morph'] = yc_obs
        return obs_params

    def generate_flux_map(self, morph_params: dict, shared_params: dict, image_shape: int, factor: int) -> jnp.ndarray:
        from jax.scipy.special import kn  # not implemented for jax<0.12.1

        amplitude = morph_params['amplitude']
        r_scale = morph_params['r_scale']
        h_scale = morph_params['h_scale']
        xc_morph = morph_params['xc_morph']
        yc_morph = morph_params['yc_morph']
        PA_morph = morph_params['PA_morph']  # degrees

        x_grid = jnp.linspace(0 - xc_morph, image_shape - xc_morph - 1, image_shape * factor)
        y_grid = jnp.linspace(0 - yc_morph, image_shape - yc_morph - 1, image_shape * factor)
        x_grid, y_grid = jnp.meshgrid(x_grid, y_grid)
        c, s = jnp.cos(jnp.radians(PA_morph)), jnp.sin(jnp.radians(PA_morph))
        rotation = jnp.array([[c, -s], [s, c]])
        r, h = jnp.einsum("ij,jhw->ihw", rotation, jnp.stack([x_grid, y_grid], axis=0))

        u = jnp.maximum(jnp.abs(r) / r_scale, 1e-10)
        model_image = amplitude * u * kn(1, u) / jnp.cosh(h / h_scale) ** 2
        return model_image

    def sample(self) -> dict:
        return self._sample()

    def sample_without_amplitude(self) -> dict:
        """Sample all parameters except amplitude (used in multi-obs fitting)."""
        from .param_spec import sample_specs
        specs = [s for s in self.parameters if s.name != 'amplitude']
        return sample_specs(specs)


MORPH_REGISTRY = {
    'Sersic': SersicMorphology,
    'EdgeDisk': EdgeDisk,
}
