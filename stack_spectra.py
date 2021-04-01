import numpy as np

from spectral_resampling import spectres


def _redshift(data_in, z_in, z_out, data_type):
    """Redshift Correction for input data

    Parameters
    ----------
    data_in : numpy.ndarray
        Input data which is either flux values, wavelengths or ivars.
        Default DESI units are assumed.
    z_in : float or numpy.ndarray
        input redshifts
    z_out : float or numpy.ndarray
        output redshifts
    data_type : str
        "flux", "wave" or "ivar"

    Returns
    -------
    numpy.ndarray
        redshift corrected value corresponding to data type
    """
    exponent_dict = {"flux": -1, "wave": 1, "ivar": 2}
    assert data_type in exponent_dict.keys(), "Not a valid Data Type"
    data_in = np.atleast_2d(data_in)

    if z_in.ndim == 1:
        z_in = z_in[:, np.newaxis]
    exponent = exponent_dict[data_type]
    data_out = data_in * ((1 + z_out) / (1 + z_in)) ** exponent

    return data_out


def _common_grid(flux, wave, ivar, z_in, z_out=0.0, wave_grid=None):
    # Correct for redshift
    z_out = np.atleast_1d(z_out)
    flux_new = _redshift(flux, z_in, z_out, "flux")
    wave_new = _redshift(wave, z_in, z_out, "wave")
    ivar_new = _redshift(ivar, z_in, z_out, "ivar")
    if wave_grid is None:
        wave_grid = np.arange(np.min(wave_new), np.max(wave_new), 0.8)
    flux_new, ivar_new = spectres(
        wave_grid, wave_new, flux_new, ivar_new, verbose=False, fill=np.nan
    )
    return flux_new, ivar_new, wave_grid