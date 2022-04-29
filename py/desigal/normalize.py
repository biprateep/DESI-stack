import numpy as np
from numpy import atleast_2d
import numpy.ma as ma

# Todo: Take desispec.Spectra as input, propagate masks, add checks.
# add various kinds of normalization, mean, median, etc.


def normalize(flux, ivar, mask=None, method="median"):
    """Normalize the flux and ivar arrays using various methods.

    Parameters
    ----------
    flux : np.ndarray or dict
        Camera coadded flux or dictionary of camera fluxes
    ivar : np.ndarray or dict
        Camera coadded inverse variance or dictionary of camera inverse variances
    mask : np.ndarray or dict, optional
        Camera coadded masks or a dictionary of camera wise masks, by default None
    method : str, optional
        Method to perform the normalization by, by default "median"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    nomalization_methods = {"median": np.median, "mean": np.mean}
    if method not in nomalization_methods:
        raise ValueError(f"Method {method} not supported")
    norm_fn = nomalization_methods[method]

    if isinstance(flux, np.ndarray):
        flux = atleast_2d(flux)
        ivar = atleast_2d(ivar)
        mask = atleast_2d(mask)
        if mask is not None:
            flux = ma.masked_array(flux, mask=mask)
        norm = norm_fn(flux, axis=-1)[:, None]
        flux_normed = flux / norm
        flux_normed = np.array(flux_normed)
        ivar_normed = np.array(ivar * norm ** 2)

    if isinstance(flux, dict):
        flux_normed = {}
        ivar_normed = {}
        for key, value in flux.items():
            if mask is not None:
                flux_normed[key] = ma.masked_array(value, mask=mask[key])
            norm = norm_fn(flux_normed[key], axis=-1)[:, None]
            flux_normed[key] = flux_normed[key] / norm
            flux_normed[key] = np.array(flux_normed[key])
            ivar_normed[key] = ivar_normed[key] * norm ** 2

    return flux_normed, ivar_normed