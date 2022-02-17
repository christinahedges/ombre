"""Tools to calibrate data"""
from typing import Optional, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling.blackbody import FLAM, BlackBody1D
from scipy.optimize import minimize

from . import PACKAGEDIR

CALIPATH = "{}{}".format(PACKAGEDIR, "/data/calibration/")


def wavelength_calibrate(visit):
    hdu = fits.open(CALIPATH + f"WFC3.IR.{visit.filter}.1st.sens.2.fits")
    x = (np.arange(visit.trace.shape[0]) / visit.trace.shape[0]) - 0.5
    data = np.copy(visit.trace) / np.median(visit.trace)
    cent = np.average(np.arange(data.shape[0]), weights=data)

    sens_raw = hdu[1].data["SENSITIVITY"]
    wav = hdu[1].data["WAVELENGTH"] * u.angstrom
    bb = BlackBody1D(visit.st_teff * u.K)(wav).to(FLAM, u.spectral_density(wav))
    bb /= np.trapz(bb, wav)
    sens = sens_raw * bb.value
    sens /= np.nanmedian(sens)

    if visit.filter == "G141":
        dw = 17000 - 10500
        meanw = (10500 + 17000) / 2
    else:
        dw = 11500 - 7700
        meanw = (11500 + 7700) / 2

    x = x * dw

    def func(params, return_model=False):
        bb = BlackBody1D(params[3] * u.K)(wav).to(FLAM, u.spectral_density(wav))
        bb /= np.trapz(bb, wav)
        sens = sens_raw * bb.value
        sens /= np.nanmedian(sens)

        model = params[2] * np.interp(
            x * params[0] + params[1] + meanw,
            wav.value,
            sens,
        )
        if return_model:
            return model
        return np.nansum((data - model) ** 2) / (np.isfinite(model).sum())

    r = minimize(
        func,
        [1, 0, 1, visit.st_teff],
        method="Powell",
        bounds=[
            (0.1, 2),
            (-1000, 1000),
            (0.4, 3),
            (visit.st_teff - 2000, visit.st_teff + 2000),
        ],
    )
    wavelength = (x * r.x[0] + r.x[1] + meanw) * u.angstrom
    sensitivity_t = func(r.x, True)
    sensitivity = np.interp(
        x * r.x[0] + r.x[1] + meanw,
        wav.value,
        sens_raw,
    )
    sensitivity_raw = sensitivity.copy()
    sensitivity /= np.median(sensitivity)
    return (
        wavelength[~visit.trace_mask],
        sensitivity[~visit.trace_mask],
        sensitivity_t[~visit.trace_mask],
        sensitivity_raw[~visit.trace_mask],
    )
