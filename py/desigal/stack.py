import sys
import numpy as np

# from scipy.interpolate import interp1d
# from scipy.ndimage import median_filter
# from scipy.stats import binned_statistic
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from spectral_resampling import resample_spectra
from . import *


# def _common_grid(flux, wave, ivar, z_in, z_out=0.0, wave_grid=None):
#     """Bring spectra to a common grid

#     Parameters
#     ----------
#     flux : np.ndarray
#         numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
#     wave : np.ndarray
#         numpy array containing the wavelength grid of shape [num_wavelengths] or [num_spectra, num_wavelengths]
#     ivar : np.ndarray
#         numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]
#     z_in : np.ndarray
#         a 1D numpy array containing the redshifts of each spectra
#     z_out : float, optional
#         common redshift for the output data, by default 0.0
#     wave_grid : np.ndarray, optional
#         a 1D vector containing the wavelength grid for the output, by default None.
#         If set to None, the wavelength grid is linearly spaced between the maximum and minimum
#         possible wavelengths after redshift correction with a bin width of 0.8 Angstrom (DESI default)

#     Returns
#     -------
#     flux_new: np.ndarray
#         All the input fluxes brought to a common redshift and wavelength grid.
#         Missing values and extrapolations are denoted with nan.
#     ivar_new: np.ndarray
#         All input inverse variances brought to a common redshift and wavelength grid.
#         Missing values and extrapolations are denoted with nan.
#     wave_grid: np.ndarray
#         The common wavelength grid.
#     """
#     # Correct for redshift
#     z_out = np.atleast_1d(z_out)
#     flux_new = _redshift(flux, z_in, z_out, "flux")
#     wave_new = _redshift(wave, z_in, z_out, "wave")
#     ivar_new = _redshift(ivar, z_in, z_out, "ivar")
#     # TODO Fix the resolution issue
#     # resample to common grid
#     if wave_grid is None:
#         wave_grid = np.arange(np.min(wave_new), np.max(wave_new), 0.32)
#     flux_new, ivar_new = resample_spectra(
#         wave_grid, wave_new, flux_new, ivar_new, verbose=False, fill=np.nan
#     )
#     return flux_new, ivar_new, wave_grid


# def _wavg(flux, ivar=None, weighted=False, weights=None):
#     """
#     Weighted average of the spectra.

#     Parameters
#     ----------
#     flux: np.ndarray
#         numpy array containing the flux grid of shape [num_spectra, num_wavelengths]

#     ivar : np.ndarray
#         numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]

#     weights : np.ndarray
#         numpy array containing the weights grid of shape [num_spectra, num_wavelengths]

#     weighted: True or False
#         if false, use weight=1 for all the spectra
#         else, perform a weighted average using the input for 'weights'

#     Returns
#     ----------
#     avg: np.ndarray
#         numpy array containing the averaged flux of shape [num_wavelengths]

#     """

#     if weighted:
#         num = np.nansum(flux * weights, axis=0)
#         denom = np.nansum(weights, axis=0)

#         if 0.0 in denom:
#             denom[denom == 0.0] = np.nan

#         avg = np.nan_to_num(num / denom)
#     else:
#         avg = np.mean(flux, axis=0)

#     ivar = np.nansum(ivar, axis=0)

#     return avg, ivar


# def _bootstrap(flux_spec, ndata, nbootstraps, len_spec):
#     """
#     Sample the spectra

#     Parameters
#     ----------
#     flux_spec:  np.ndarray
#         Numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
#         To avoid redundant calculations, this array can be the normalized spectra already
#         brought to the common grid

#     ndata: int
#         The number of spectra to sample from

#     nbootstraps: int
#         The number of times to sample the data

#     nsamples: int
#         The number of bootstraps to do

#     len_spec: int
#         The number of wavelengths in the spectra

#     Returns
#     ----------
#     stacks: np.ndarray
#         numpy array containing the stacked spectra from the bootstraps of size [nsamples, len_spec]

#     ivar: np.ndarray
#         numpy array of size [len_spec] containing the inverse variance calculated from all the stacks

#     """

#     boot = np.zeros((nbootstraps, len_spec))
#     for i in range(nbootstraps):
#         idx = np.random.choice(ndata, 1, replace=True)
#         boot[i] += flux_spec[idx][0]

#     return boot


# def stack_spectra(flux, wave, ivar, sky=None, bootstrap=False):
#     """
#     If flux/wave.ivar are dicts, coadd cameras.
#     If sky is present model ivar and do modelled ivar weighted avg
#     """

#     stacks = np.zeros((nbootstraps, len_spec))
#     for j in range(nbootstraps):
#         boot = np.zeros((nsamples, len_spec))
#         for i in range(nsamples):
#             idx = np.random.choice(ndata, 1, replace=True)
#             boot[i] += flux_spec[idx][0]
#         stacks[j] += wavg(boot)

#     ivar = 1.0 / (np.nanstd(stacks, axis=0)) ** 2

#     return stacks, ivar


# def model_ivar(ivar, sky_ivar, wave):
#     n_obj = len(sky_ivar)
#     sky_var = 1 / sky_ivar

#     ivar_model = np.zeros_like(ivar)

#     for i in range(n_obj):
#         sky_mask = np.isfinite(sky_var[i])
#         sky_var_interp = interp1d(
#             wave[sky_mask], sky_var[i][sky_mask], fill_value="extrapolate", axis=-1
#         )
#         sky_var[i] = sky_var_interp(wave)
#         sky_var[i] = sky_var[i] / median_filter(
#             sky_var[i], 100
#         )  # takes out the overall shape of sky var

#         # Create polunomial function of wavelength
#         poly_feat_m = PolynomialFeatures(3)
#         poly_feat_c = PolynomialFeatures(3)
#         coef_m = poly_feat_m.fit_transform(wave[:, np.newaxis])
#         coef_c = poly_feat_c.fit_transform(wave[:, np.newaxis])

#         obj_var = 1 / (ivar[i])
#         obj_mask = np.isfinite(obj_var)  # TODO Check for Nan values here
#         obj_back = median_filter(obj_var[obj_mask], 100, mode="nearest")
#         X = (
#             np.concatenate(
#                 [(coef_m * sky_var[i][:, np.newaxis])[obj_mask], coef_c[obj_mask]],
#                 axis=1,
#             )
#             + obj_back[:, np.newaxis]
#         )
#         Y = obj_var[obj_mask]
#         model = LinearRegression(fit_intercept=False, n_jobs=-1)
#         model.fit(X, Y)
#         y_predict = model.predict(X)
#         residual = (Y - y_predict) / Y
#         # correct for the overall shape of the residuals
#         wave_bins = np.arange(wave.min(), wave.max(), 500)
#         binned_residual, _, _ = binned_statistic(
#             wave[obj_mask], residual, statistic="median", bins=wave_bins
#         )
#         interp_binned_res = interp1d(
#             (wave_bins[1:] + wave_bins[:-1]) / 2,
#             binned_residual,
#             kind="cubic",
#             fill_value="extrapolate",
#         )
#         large_res = interp_binned_res(wave[obj_mask])
#         y_pred_adjust = large_res * Y + y_predict
#         ivar_model[i][obj_mask] = 1 / y_pred_adjust
#         ivar_model[i][~obj_mask] = 0
#     return ivar_model


def stack_spectra(
    flux,
    wave,
    ivar,
    mask,
    spec_z,
    fibermap,
    exp_fibermap,
    resample_resolution=0.32,
    norm_method="median",
    resample_method="linear",
    weight="none",
    stack_method="mean",
    stack_error="bootstrap",
):

    flux_dered = redshift(flux, spec_z, 0, "flux")
    wave_dered = redshift(wave, spec_z, 0, "wave")
    ivar_dered = redshift(ivar, spec_z, 0, "ivar")

    flux_coadd, wave_coadd, ivar_coadd, mask_coadd = coadd_cameras(
        flux_dered, wave_dered, ivar_dered, mask
    )
    flux_normed, ivar_normed = normalize(
        flux_coadd, ivar_coadd, mask=mask, method=norm_method
    )
    wave_grid = np.arange(np.min(wave_coadd), np.max(wave_coadd), resample_resolution)
    flux_grid, ivar_grid = resample(
        wave_grid,
        wave_coadd,
        flux_normed,
        ivar_normed,
        verbose=False,
        fill=np.nan,
        method="linear",
    )

    stacked_spectra = np.mean(flux_grid, axis=0)

    return stacked_spectra, wave_grid