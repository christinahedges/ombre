"""Common ombre methods"""
import warnings

import astropy.units as u
import exoplanet as xo
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pymc3 as pm
import theano.tensor as tt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.modeling.blackbody import blackbody_lambda
from astropy.stats import sigma_clipped_stats
from fbpca import pca
from matplotlib import animation
from numpy import ma
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import pearsonr
from tqdm import tqdm

from . import PACKAGEDIR
from .spec import Spectra, Spectrum

# from numba import jit


CALIPATH = "{}{}".format(PACKAGEDIR, "/data/calibration/")


def animate(data, scale="linear", output="out.mp4", **kwargs):
    """Create an animation of all the frames in `data`.

    Parameters
    ----------
    data : np.ndarray
        3D np.ndarray
    output : str
        File to output mp4 to
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    idx = 0
    if scale is "log":
        dat = np.log10(np.copy(data))
    else:
        dat = data
    cmap = kwargs.pop("cmap", "Greys_r")
    cmap = plt.get_cmap(cmap)
    cmap.set_bad("black")
    if "vmax" not in kwargs:
        kwargs["vmin"] = np.nanpercentile(dat, 70)
        kwargs["vmax"] = np.nanpercentile(dat, 99.9)
    im = ax.imshow(dat[idx], origin="lower", cmap=cmap, **kwargs)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis("off")

    def animate(idx):
        im.set_data(dat[idx])
        return (im,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(dat), interval=30, blit=True
    )
    anim.save(output, dpi=150)


def simple_mask(
    sci: npt.NDArray[np.float64],
    target_block: npt.NDArray[np.float64],
    filter: str = "G141",
) -> (npt.NDArray[bool], npt.NDArray[np.float64], npt.NDArray[np.float64]):
    """Build a 4D boolean mask that is true where the spectrum is on the detector.

    Parameters
    ----------
    sci : np.ndarray
        4D data cube of each ramp, frame, column and row.

    Returns
    -------
    mask : np.ndarray of bools
        4D mask, true where the spectrum is dispersed on the detector.
    spectral : np.ndarray of floats
        The average data in the spectral dimension
    spatial : np.ndarray of floats
        The average of the data in the spatial dimension.
    """
    mask = np.atleast_3d(sci.mean(axis=0) > np.nanpercentile(sci, 50)).transpose(
        [2, 0, 1]
    )
    for count in [0, 1]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = sci / mask
            data[~np.isfinite(data)] = np.nan
            spectral = np.nanmean(data, axis=(0, 1))
            spatial = np.nanmean(data, axis=(0, 2))

        # These are hard coded to be generous.
        spatial_cut = 0.4
        if filter == "G102":
            spectral_cut = 0.4
        if filter == "G141":
            spectral_cut = 0.3

        spectral = spectral > np.nanmax(spectral) * spectral_cut
        # G102 edge is hard to find.
        # if filter == "G102":
        #     edge = (
        #         np.where((np.gradient(spectral / np.nanmax(spectral)) < -0.07))[0][0] - 10
        #     )
        #     m = np.ones(len(spectral), bool)
        #     m[edge:] = False
        #     spectral &= m

        spatial = spatial > np.nanmax(spatial) * spatial_cut

        mask = (
            np.atleast_3d(np.atleast_2d(spectral) * np.atleast_2d(spatial).T)
        ).transpose([2, 0, 1])

        # Cut out the zero phase dispersion!
        # Choose the widest block as the true dispersion...
        secs = mask[0].any(axis=0).astype(int)
        sec_l = np.hstack(
            [0, np.where(np.gradient(secs) != 0)[0][::2] + 1, mask.shape[1]]
        )
        dsec_l = np.diff(sec_l)
        msec_l = np.asarray([s.mean() for s in np.array_split(secs, sec_l[1:-1])])
        l = sec_l[np.argmax(dsec_l * msec_l) : np.argmax(dsec_l * msec_l) + 2]
        m = np.zeros(mask.shape[1:], bool)
        m[:, l[0] : l[1]] = True
        mask[0] &= m

        secs = mask[0].any(axis=1).astype(int)
        sec_l = np.hstack(
            [0, np.where(np.gradient(secs) != 0)[0][::2] + 1, mask.shape[1]]
        )
        dsec_l = np.diff(sec_l)
        msec_l = np.asarray([s.mean() for s in np.array_split(secs, sec_l[1:-1])])
        if (msec_l == 1).sum() == 1:
            break
        else:
            mask = np.atleast_3d(
                sci.mean(axis=0) > np.nanpercentile(sci, 50)
            ).transpose([2, 0, 1])
            mask[0] &= target_block

    spectral, spatial = (
        np.atleast_3d(np.atleast_2d(spectral & mask[0].any(axis=0))).transpose(
            [2, 0, 1]
        ),
        np.atleast_3d(np.atleast_2d((spatial & mask[0].any(axis=1)).T)).transpose(
            [2, 1, 0]
        ),
    )
    if spatial.sum() < 4:
        while spatial.sum() < 4:
            spatial |= (np.gradient(spatial[0, :, 0].astype(float)) != 0)[None, :, None]
        mask = spatial & spectral

    return (mask, spectral, spatial)
