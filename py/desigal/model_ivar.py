import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import binned_statistic


def model_ivar(ivar, sky, wave, mask=None):
    n_obj = len(sky)

    ivar_model = np.zeros_like(ivar)

    for i in range(n_obj):
        sky_mask = np.isfinite(sky[i])
        sky_interp = interp1d(
            wave[sky_mask], sky[i][sky_mask], fill_value="extrapolate", axis=-1
        )
        sky[i] = sky_interp(wave)
        # sky[i] = sky[i]/median_filter(sky[i], 100) #takes out the overall shape of sky var

        # Create polynomial function of wavelength
        poly_feat_m = PolynomialFeatures(3)
        poly_feat_c = PolynomialFeatures(3)
        coef_m = poly_feat_m.fit_transform(wave[:, np.newaxis])
        coef_c = poly_feat_c.fit_transform(wave[:, np.newaxis])

        obj_var = 1 / np.sqrt(ivar[i])
        obj_mask = np.isfinite(obj_var)  # TODO Check for Nan values here
        obj_back = median_filter(obj_var[obj_mask], 200, mode="nearest")
        X = (
            np.concatenate(
                [(coef_m * sky[i][:, np.newaxis])[obj_mask], coef_c[obj_mask]], axis=1
            )
            + obj_back[:, np.newaxis]
        )
        Y = obj_var[obj_mask]
        model = LinearRegression(fit_intercept=False, n_jobs=-1)
        model.fit(X, Y)
        y_predict = model.predict(X)
        residual = (Y - y_predict) / Y
        # correct for the overall shape of the residuals
        wave_bins = np.arange(wave.min(), wave.max(), 400)
        binned_residual, _, _ = binned_statistic(
            wave[obj_mask], residual, statistic="median", bins=wave_bins
        )
        interp_binned_res = interp1d(
            (wave_bins[1:] + wave_bins[:-1]) / 2,
            binned_residual,
            kind="cubic",
            fill_value="extrapolate",
        )
        large_res = interp_binned_res(wave[obj_mask])
        y_pred_adjust = large_res * Y + y_predict
        ivar_model[i][obj_mask] = 1 / y_pred_adjust ** 2
        ivar_model[i][~obj_mask] = 0

    return ivar_model