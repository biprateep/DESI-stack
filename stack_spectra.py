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
    """Bring spectra to a common grid

    Parameters
    ----------
    flux : np.ndarray
        numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
    wave : np.ndarray
        numpy array containing the wavelength grid of shape [num_wavelengths] or [num_spectra, num_wavelengths]
    ivar : np.ndarray
        numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]
    z_in : np.ndarray
        a 1D numpy array containing the redshifts of each spectra
    z_out : float, optional
        common redshift for the output data, by default 0.0
    wave_grid : np.ndarray, optional
        a 1D vector containing the wavelength grid for the output, by default None.
        If set to None, the wavelength grid is linearly spaced between the maximum and minimum
        possible wavelengths after redshift correction with a bin width of 0.8 Angstrom (DESI default)

    Returns
    -------
    flux_new: np.ndarray
        All the input fluxes brought to a common redshift and wavelength grid.
        Missing values and extrapolations are denoted with nan.
    ivar_new: np.ndarray
        All input inverse variances brought to a common redshift and wavelength grid.
        Missing values and extrapolations are denoted with nan.
    wave_grid: np.ndarray
        The common wavelength grid.
    """
    # Correct for redshift
    z_out = np.atleast_1d(z_out)
    flux_new = _redshift(flux, z_in, z_out, "flux")
    wave_new = _redshift(wave, z_in, z_out, "wave")
    ivar_new = _redshift(ivar, z_in, z_out, "ivar")
    # resample to common grid
    if wave_grid is None:
        wave_grid = np.arange(np.min(wave_new), np.max(wave_new), 0.8)
    flux_new, ivar_new = spectres(
        wave_grid, wave_new, flux_new, ivar_new, verbose=False, fill=np.nan
    )
    return flux_new, ivar_new, wave_grid


def _normalize(flux,ivar):
    """
    A simple normalization to median=1 for flux
    Also adjusts inverse variance accordingly
    
    Parameters
    ----------
    flux: np.ndarray 
        numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
        
    ivar : np.ndarray
        numpy array containing the inverse variance grid of shape [num_spectra, num_wavelengths]
    
    Returns
    -------
    flux_new: np.ndarray
        flux that has been normalized to median one
        
    ivar_new: np.ndarray
        inverse variance that has been multipled by the normalization factor
        for the flux,squared

    """
    
    norm = np.nanmedian(flux,axis=1).reshape(315,1)
    flux = flux/norm
    ivar = ivar*norm**2
    
    return flux, ivar

def _wavg(flux,weights=None,weighted=False):
    """
    Weighted average of the spectra.
    
    Parameters
    ----------
    flux: np.ndarray 
        numpy array containing the flux grid of shape [num_spectra, num_wavelengths]
        
    weights : np.ndarray
        numpy array containing the weights grid of shape [num_spectra, num_wavelengths]
    
    weighted: True or False
        if false, use weight=1 for all the spectra
        else, perform a weighted average using the input for 'weights'
    
    """
    
    if weighted:
        num = np.nansum(flux*weights,axis=0)
        denom = np.nansum(weights,axis=0)
    
        if 0. in denom:
            denom[denom==0.0] = np.nan
    
        avg = np.nan_to_num(num/denom)
    else:
        avg = np.mean(flux,axis=0)
    return(avg)