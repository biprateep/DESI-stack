import numpy as np

# Todo: Take desispec.Spectra as input, propagate masks, add checks.
# add various kinds of normalization, mean, median, etc.
def normalize(flux, ivar):
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
    flux: np.ndarray
        flux that has been normalized to median one

    ivar: np.ndarray
        inverse variance that has been multipled by the normalization factor
        for the flux,squared

    """

    norm = np.nanmedian(flux, axis=1, keepdims=True)
    flux = flux / norm
    ivar = ivar * norm ** 2

    return flux, ivar