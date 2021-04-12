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
    data_out = data_in * (((1 + z_out) / (1 + z_in)) ** exponent)

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


def coadd_cameras(spectra, cosmics_nsig=0.0):

    # check_alignement_of_camera_wavelength(spectra)

    log = get_logger()

    # ordering
    mwave = [np.mean(spectra.wave[b]) for b in spectra.bands]
    sbands = np.array(spectra.bands)[
        np.argsort(mwave)
    ]  # bands sorted by inc. wavelength
    log.debug("wavelength sorted cameras= {}".format(sbands))

    # create wavelength array
    wave = None
    tolerance = 0.0001  # A , tolerance
    for b in sbands:
        if wave is None:
            wave = spectra.wave[b]
        else:
            wave = np.append(
                wave, spectra.wave[b][spectra.wave[b] > wave[-1] + tolerance]
            )
    nwave = wave.size

    # check alignment
    number_of_overlapping_cameras = np.zeros(nwave)
    for b in spectra.bands:
        windices = np.argmin(
            (
                np.tile(wave, (spectra.wave[b].size, 1))
                - np.tile(spectra.wave[b], (wave.size, 1)).T
            )
            ** 2,
            axis=1,
        )
        dist = np.sqrt(np.max(spectra.wave[b] - wave[windices]))
        log.debug("camera {} max dist= {}A".format(b, dist))
        if dist > tolerance:
            log.error(
                "Cannot directly coadd the camera spectra because wavelength are not aligned, use --lin-step or --log10-step to resample to a common grid"
            )
            sys.exit(12)
        number_of_overlapping_cameras[windices] += 1

    # targets
    targets = np.unique(spectra.fibermap["TARGETID"])
    ntarget = targets.size
    log.debug("number of targets= {}".format(ntarget))

    # ndiag = max of all cameras
    ndiag = 0
    for b in sbands:
        ndiag = max(ndiag, spectra.resolution_data[b].shape[1])
    log.debug("ndiag= {}".format(ndiag))

    b = sbands[0]
    flux = np.zeros((ntarget, nwave), dtype=spectra.flux[b].dtype)
    ivar = np.zeros((ntarget, nwave), dtype=spectra.ivar[b].dtype)
    if spectra.mask is not None:
        ivar_unmasked = np.zeros((ntarget, nwave), dtype=spectra.ivar[b].dtype)
        mask = np.zeros((ntarget, nwave), dtype=spectra.mask[b].dtype)
    else:
        ivar_unmasked = ivar
        mask = None

    rdata = np.zeros((ntarget, ndiag, nwave), dtype=spectra.resolution_data[b].dtype)

    for b in spectra.bands:
        log.debug("coadding band '{}'".format(b))

        # indices
        windices = np.argmin(
            (
                np.tile(wave, (spectra.wave[b].size, 1))
                - np.tile(spectra.wave[b], (wave.size, 1)).T
            )
            ** 2,
            axis=1,
        )

        band_ndiag = spectra.resolution_data[b].shape[1]

        fiberstatus_bits = get_all_fiberbitmask_with_amp(b)
        good_fiberstatus = (spectra.fibermap["FIBERSTATUS"] & fiberstatus_bits) == 0
        for i, tid in enumerate(targets):
            jj = np.where((spectra.fibermap["TARGETID"] == tid) & good_fiberstatus)[0]

            # - if all spectra were flagged as bad (FIBERSTATUS != 0), contine
            # - to next target, leaving tflux and tivar=0 for this target
            if len(jj) == 0:
                continue

            ivar_unmasked[i, windices] += np.sum(spectra.ivar[b][jj], axis=0)

            if spectra.mask is not None:
                ivarjj = spectra.ivar[b][jj] * (spectra.mask[b][jj] == 0)
            else:
                ivarjj = spectra.ivar[b][jj]

            ivar[i, windices] += np.sum(ivarjj, axis=0)
            flux[i, windices] += np.sum(ivarjj * spectra.flux[b][jj], axis=0)
            for r in range(band_ndiag):
                rdata[i, r + (ndiag - band_ndiag) // 2, windices] += np.sum(
                    (spectra.ivar[b][jj] * spectra.resolution_data[b][jj, r]), axis=0
                )
            if spectra.mask is not None:
                # this deserves some attention ...

                tmpmask = np.bitwise_and.reduce(spectra.mask[b][jj], axis=0)

                # directly copy mask where no overlap
                jj = number_of_overlapping_cameras[windices] == 1
                mask[i, windices[jj]] = tmpmask[jj]

                # 'and' in overlapping regions
                jj = number_of_overlapping_cameras[windices] > 1
                mask[i, windices[jj]] = mask[i, windices[jj]] & tmpmask[jj]

    for i, tid in enumerate(targets):
        ok = ivar[i] > 0
        if np.sum(ok) > 0:
            flux[i][ok] /= ivar[i][ok]
        ok = ivar_unmasked[i] > 0
        if np.sum(ok) > 0:
            rdata[i][:, ok] /= ivar_unmasked[i][ok]

    if "COADD_NUMEXP" in spectra.fibermap.colnames:
        fibermap = spectra.fibermap
    else:
        fibermap = coadd_fibermap(spectra.fibermap)

    bands = ""
    for b in sbands:
        bands += b

    if spectra.mask is not None:
        dmask = {
            bands: mask,
        }
    else:
        dmask = None

    res = Spectra(
        bands=[
            bands,
        ],
        wave={
            bands: wave,
        },
        flux={
            bands: flux,
        },
        ivar={
            bands: ivar,
        },
        mask=dmask,
        resolution_data={
            bands: rdata,
        },
        fibermap=fibermap,
        meta=spectra.meta,
        extra=spectra.extra,
        scores=None,
    )

    if spectra.scores is not None:
        orig_scores = spectra.scores.copy()
        orig_scores["TARGETID"] = spectra.fibermap["TARGETID"]
    else:
        orig_scores = None

    compute_coadd_scores(res, orig_scores, update_coadd=True)

    return res


def _normalize(flux, ivar):
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

    norm = np.nanmedian(flux, axis=1, keepdims=True)
    flux = flux / norm
    ivar = ivar * norm ** 2

    return flux, ivar


def _wavg(flux, weights=None, weighted=False):
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

    Returns
    ----------
    avg: np.ndarray
        numpy array containing the averaged flux of shape [num_wavelengths]

    """

    if weighted:
        num = np.nansum(flux * weights, axis=0)
        denom = np.nansum(weights, axis=0)

        if 0.0 in denom:
            denom[denom == 0.0] = np.nan

        avg = np.nan_to_num(num / denom)
    else:
        avg = np.mean(flux, axis=0)
    return avg


def _bootstrap(flux, wave, ivar, z_in, z_out=0.0, wave_grid=None, ndata, nbootstraps, len_spec):
    """
    Sample the spectra

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

    ndata: int
        The number of spectra to sample from

    nbootstraps: int
        The number of times to sample the data

    len_spec: int
        The number of wavelengths in the spectra

    Returns
    ----------
    boot: np.ndarray
        numpy array containing the sampled spectra of shape [nbootsraps, len_spec]

    """
    boot = np.zeros((nbootstraps, len_spec))
    for i in range(nbootstraps):
        idx = np.random.choice(ndata, 1, replace=True)
        boot[i] += flux_spec[idx][0]

    return boot
