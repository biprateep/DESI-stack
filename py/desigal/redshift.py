import numpy as np

# Todo: Take desispec.Spectra as input, propagate masks, add checks.


def redshift(data_in, z_in, z_out, data_type):
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
    data_out = data_in * (((1 + z_out) / (1 + z_in)) ** exponent)

    return data_out