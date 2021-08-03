"""Tools to calibrate data"""
import numpy as np
from astropy.modeling.blackbody import BlackBody1D, FLAM
from . import PACKAGEDIR
from scipy.optimize import minimize
import astropy.units as u
from typing import Optional, Union
from astropy.io import fits
import matplotlib.pyplot as plt

from scipy.optimize import minimize

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
    sensitivity /= np.median(sensitivity)
    return (
        wavelength[~visit.trace_mask],
        sensitivity[~visit.trace_mask],
        sensitivity_t[~visit.trace_mask],
    )


#
# def wavelength_calibrate(
#     visit, teff: Optional[Union[int, float]] = 6000, plot: Optional[bool] = False
# ):
#     """
#     Provides wavelength calibration for visit data
#
#     Parameters
#     ----------
#     visit: ombre.visit.Visit object
#         The visit to provide a wavelength calibration for
#     teff: float
#         Optional temperature of target to estimate the blackbody
#     plot: bool
#         Optional keyword argument to produce a diagnostic plot
#
#     Returns
#     -------
#     wavelength : np.ndarray
#         Best fit wavelength array
#     sensitivity: np.ndarray
#         Array of best fit sensitivity curve
#     sensitivity_t: np.ndarray
#         Array of best fit sensitivity curve multiplied by best fit blackbody curve
#
#     """
#     hdu = fits.open(CALIPATH + f"WFC3.IR.{visit.filter}.1st.sens.2.fits")
#     data = np.copy(visit.trace) / np.median(visit.trace)
#     cent = np.average(np.arange(data.shape[0]), weights=data)
#
#     for count in [0, 1, 2]:
#         sens_raw = hdu[1].data["SENSITIVITY"]
#         wav = hdu[1].data["WAVELENGTH"] * u.angstrom
#         bb = BlackBody1D(teff * u.K)(wav).to(FLAM, u.spectral_density(wav))
#         bb /= np.trapz(bb, wav)
#         sens = sens_raw * bb.value
#         sens /= np.nanmedian(sens)
#
#         if visit.filter == "G141":
#             dw = 17000 - 10500
#         else:
#             dw = 11500 - 7700
#         w_mean = np.mean(wav.value)
#
#         # g1 = np.gradient(sens)
#         # g1[:100] = 0
#         # g1[-100:] = 0
#         # if visit.filter == "G141":
#         #     sens[np.argmax(g1) + 300 : np.argmin(g1) - 300] = np.nan
#
#         def func(params, return_model=False):
#             model = np.interp(
#                 np.arange(0, data.shape[0]),
#                 (wav.value - w_mean) / dw * params[0] + params[1],
#                 sens,
#             )
#             model *= np.median(data / model)
#             if return_model:
#                 return model
#             return np.nansum((data - model) ** 2) / (np.isfinite(model).sum())
#
#         na = 100
#         nb = 101
#         if visit.filter == "G141":
#             a = np.linspace((data.shape[0]) * 0.85, (data.shape[0]) * 1.1, na)
#             b = np.linspace(cent - 10, cent + 10, nb)
#         if visit.filter == "G102":
#             a = np.linspace((data.shape[0]) * 0.85, (data.shape[0]) * 1.1, na)
#             b = np.linspace(cent - 7, cent + 7, nb)
#
#         chi = np.zeros((na, nb))
#         for idx, a1 in enumerate(a):
#             for jdx, b1 in enumerate(b):
#                 model = func([a1, b1], return_model=True)
#                 if visit.filter == "G102":
#                     model[
#                         np.where(data > 0.6)[0][0] : np.where(np.gradient(data) < -0.1)[
#                             0
#                         ][0]
#                         - 5
#                     ] = np.nan
#                 chi[idx, jdx] = np.nansum((data - model) ** 2) / (
#                     np.isfinite(model).sum()
#                 )
#         l = np.unravel_index(np.argmin(chi), chi.shape)
#         params = [a[l[0]], b[l[1]]]
#         wavelength = (
#             ((np.arange(data.shape[0]) - params[1]) / params[0]) * dw + w_mean
#         ) * u.angstrom
#         sens = sens_raw  # * bb.value[10:-10]
#         sens /= np.median(sens)
#         sensitivity = np.interp(wavelength.value, wav.value, sens)
#
#         def func(teff):
#             bb = (
#                 BlackBody1D(teff * u.K)(wavelength).to(
#                     FLAM, u.spectral_density(wavelength)
#                 )
#             ).value
#             bb /= np.trapz(bb, wavelength.value)
#             sens1 = sensitivity * bb
#             sens1 /= np.median(sens1)
#             return np.nansum((data - sens1) ** 2 / data)
#
#         r = minimize(func, [teff], method="TNC", bounds=[(1000, 20000)])
#         teff = r.x[0]
#
#     bb = (
#         BlackBody1D(teff * u.K)(wavelength).to(FLAM, u.spectral_density(wavelength))
#     ).value
#     bb /= np.trapz(bb, wavelength.value)
#     sensitivity_t = sensitivity * bb
#     sensitivity_t /= np.median(sensitivity_t)
#
#     if plot:
#         plt.figure()
#         plt.pcolormesh(a, b, chi.T, cmap="coolwarm")
#         plt.colorbar()
#         plt.scatter(*params, c="lime")
#         plt.scatter(data.shape[0], cent, c="orange", marker="x")
#     return (
#         wavelength[~visit.trace_mask],
#         sensitivity[~visit.trace_mask],
#         sensitivity_t[~visit.trace_mask],
#     )
