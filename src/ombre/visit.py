"""Class for handling visit data"""
import warnings
from datetime import datetime
from glob import glob
from typing import Generic, List, Optional, Tuple, TypeVar, Union
from urllib.request import URLError

from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.convolution import Box2DKernel, convolve
from astropy.io import fits, votable
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from dataclasses import dataclass
from lightkurve.units import ppm
from scipy import sparse

from . import PACKAGEDIR
from .calibrate import wavelength_calibrate
from .matrix import vstack, vstack_independent
from .query import get_nexsci
from .spec import Spectra, Spectrum
from .utils import simple_mask

CALIPATH = "{}{}".format(PACKAGEDIR, "/data/calibration/")


def _auto_mask(x: npt.NDArray, mask: npt.NDArray[bool], dim: Optional[int] = None):
    """Masks an array of an arbitrary shape with a 1D boolean mask

    Parameters
    ----------
    x : np.ndarray of any shape
        Input array to mask
    mask : np.ndarray of boolean
        Array of booleans
    dim : Optional, int
        Dimension on which to mask. If not provided, will find an axis with the same length as `mask`.
        If more than one exists, will raise an error.

    Returns
    -------
    x_masked : np.ndarray
        Masked array
    """
    if dim is None:
        l = np.where(np.asarray(x.shape) == mask.shape[0])[0]
        if len(l) != 1:
            raise ValueError("multiple axes could be masked")
        l = l[0]
    if dim is not None:
        l = dim % x.ndim
    l = np.hstack([l, list(set(np.arange(x.ndim)) - set([l]))]).astype(int)
    return x.transpose(l)[mask].transpose(np.argsort(l))


@dataclass
class Visit(object):
    """Class for keeping visit level data"""

    sci: npt.NDArray[np.float64]
    err: npt.NDArray[np.float64]
    dq: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    exptime: float
    scan_length: float
    cadence_mask: Optional[npt.NDArray[bool]] = None
    name: Optional[str] = None
    planets: Optional[str] = None
    forward: bool = True
    visit_number: Optional[int] = 1
    propid: Optional[int] = 0
    name: Optional[str] = None
    filter: Optional[str] = "G141"
    filenames: Optional[List[str]] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    wcs: Optional = None

    def __post_init__(self):
        """Runs after init"""
        self.ns = self.sci.shape[1]
        if self.ns == 1024:
            self.ns = 1014
        self.mask, self.spectral, self.spatial = simple_mask(
            self.sci, target_block=self._target_block, filter=self.filter
        )
        self.flat = np.ones((1, *self.sci.shape[1:]))
        for count in [0, 1]:
            self.data = (self.sci / self.flat)[:, self.spatial.reshape(-1)][
                :, :, self.spectral.reshape(-1)
            ]
            self.error = (self.err / self.flat)[:, self.spatial.reshape(-1)][
                :, :, self.spectral.reshape(-1)
            ]
            self.error[self.error / self.data > 0.1] = 1e10
            self.shape = self.data.shape
            self.nt, self.nsp, self.nwav = self.shape
            self.model = self._basic_spectrum * self._basic_vsr
            self.model /= np.atleast_3d(self.model.mean(axis=(1, 2))).transpose(
                [1, 0, 2]
            )
            self.model_err = np.zeros_like(self.model)
            if count == 0:
                self.flat = self.get_flatfield()

        # Large gradient residuals
        bad_pixel_mask = self.error / self.data > 0.1
        grad = (np.asarray(np.gradient(self.data)) ** 2).sum(axis=0)
        grad = np.ma.masked_array(grad, bad_pixel_mask)
        for tdx in range(self.nt):
            bad_pixel_mask[tdx] |= sigma_clip(grad[tdx], axis=(1), sigma=6).mask
        self.error[bad_pixel_mask] = 1e10

        T = (
            np.atleast_3d(self.time) * np.ones(self.shape).transpose([1, 0, 2])
        ).transpose([1, 0, 2])
        T -= self.time[0]
        self.T = T / T.max() - 0.5

        Y, X = np.mgrid[: self.shape[1], : self.shape[2]]
        Y = Y / (self.shape[1] - 1) - 0.5
        X = X / (self.shape[2] - 1) - 0.5

        self.X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.shape)
        self.Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(self.shape)

        # self.mask_cosmics()

        if self.cadence_mask == None:
            cadence_mask = np.ones(self.nt, bool)

        # self.model_lc = self.sys.flux(t=self.time).eval()
        # 6 sigmal residuals
        self.model = (
            self.average_spectrum * self.average_vsr  # * self.average_lc[:, None, None]
        )
        bad_pixel_mask = sigma_clip(
            np.ma.masked_array(self.basic_residuals, self.error / self.data > 0.1),
            sigma=6,
            cenfunc=lambda x, axis: 0,
        ).mask
        self.error[bad_pixel_mask] = 1e10
        self._build_regressors()
        self._subtime = np.hstack(
            [
                self.time - (self.exptime * u.second.to(u.day)) / 2,
                self.time + (self.exptime * u.second.to(u.day)) / 2,
            ]
        )
        self.A = self._prepare_design_matrix()

    @property
    def _target_block(self):
        def block(parity):
            x0, y0 = self.wcs.all_world2pix([[self.ra, self.dec]], 0)[0]
            x1, x2 = x0 + 10, x0 + 200
            ys = y0, y0 - float((-1) ** (parity)) * self.scan_length * 9.5
            y1 = np.min(ys) - 30
            y2 = np.max(ys) + 30
            Y, X = np.mgrid[: self.ns, : self.ns]
            return (X > x1) & (X < x2) & (Y > y1) & (Y < y2)

        parity = np.argmax([(self.sci * block(0)).sum(), (self.sci * block(1)).sum()])
        return block(parity)

    def __repr__(self):
        return "{} Visit {}, Forward {} [Proposal ID: {}]".format(
            self.name, self.visit_number, self.forward, self.propid
        )

    def calibrate(self):
        if not hasattr(self, "st_teff"):
            get_nexsci(self, self.planets)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (
                self.wavelength,
                self.sensitivity,
                self.sensitivity_t,
                self.sensitivity_raw,
            ) = wavelength_calibrate(self)

        if self.filter == "G141":
            l = np.where(np.abs(np.gradient(self.sensitivity_t)) > 0.06)[0]
            lower, upper = (
                l[np.argmax(np.diff(l))] + 3,
                l[np.argmax(np.diff(l)) + 1] - 5,
            )
            if lower > 20:
                lower = 0
            if upper < (self.sensitivity_t.shape[0] - 20):
                upper = self.sensitivity_t.shape[0]
        else:
            lower = 0
            upper = self.sensitivity_t.shape[0]
        if (upper - lower) <= 5:
            raise ValueError(f"Can not find trace: {lower} {upper}")
        self.trace_wavelength = self.wavelength.copy()
        # Mask out edges of trace
        attrs = [
            "data",
            "error",
            "model",
            "model_err",
            "T",
            "X",
            "Y",
            "sensitivity",
            "sensitivity_raw",
            "wavelength",
        ]

        mask = np.in1d(np.arange(self.nwav), np.arange(lower, upper))
        [
            setattr(self, attr, _auto_mask(getattr(self, attr), mask, dim=-1))
            for attr in attrs
        ]
        self.shape = self.data.shape
        self.nt, self.nsp, self.nwav = self.shape

    def _build_regressors(self):
        """Builds xshift, yshift, and bkg regressors for fitting transit"""
        w = np.ones(self.sci.shape)
        w[self.err / self.sci > 0.1] = 1e10

        larger_mask = convolve(self.mask[0], Box2DKernel(31)) > 1e-5
        if larger_mask[:, -1].sum() != 0:
            larger_mask[:, -5:] |= False

        thumb = np.average(self.sci / self.flat, weights=self.flat / self.err, axis=0)
        l = np.where(larger_mask.any(axis=0))[0]
        thumb = thumb[:, l[0] : l[-1] + 1]
        l = np.where(larger_mask.any(axis=1))[0]
        thumb = thumb[l[0] : l[-1] + 1, :]

        self.trace = np.mean(thumb, axis=0)
        self.trace_mask = (larger_mask.any(axis=0) & ~self.mask[0].any(axis=0))[
            larger_mask.any(axis=0)
        ]

        Y, X = np.mgrid[: self.sci.shape[1], : self.sci.shape[2]]
        X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.sci.shape)
        Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(self.sci.shape)
        w1 = (self.sci / w)[:, larger_mask]
        w1 = (w1.T / w1.mean(axis=1)).T
        xshift = np.mean(X[:, larger_mask] * w1, axis=1)
        self.xshift = xshift - np.median(xshift)
        yshift = np.mean(Y[:, larger_mask] * w1, axis=1)
        self.yshift = yshift - np.median(yshift)

        masked_data = np.ma.masked_array(
            self.sci,
            ~larger_mask[None, :, :] * np.ones(self.sci.shape, bool)
            & (self.sci[0] == 0),
        )

        bkg = sigma_clipped_stats(
            masked_data, axis=(1, 2), sigma=3, cenfunc=np.mean, maxiters=3, grow=1.5
        )[1]
        self.bkg = bkg - np.median(bkg)

    @staticmethod
    def from_files(
        filenames=List[str],
        visit_number: int = 1,
        forward: bool = True,
        t0: Optional[float] = None,
        period: Optional[float] = None,
        force=False,
        pixel_mask=None,
    ):
        """Create a visit from multiple files

        Parameters
        ----------
        filenames : list of str
            List of file names to create visit from
        visit_number : int
            The visit number in the total observation
        forward : bool
            Whether to use only forward or backward scans
        """
        if isinstance(filenames, str):
            if not filenames.endswith("*"):
                filenames = np.asarray(glob(filenames + "*"))
            else:
                filenames = np.asarray(glob(filenames))
        else:
            filenames = np.asarray(filenames)
        if len(filenames) == 0:
            raise ValueError("Must pass >0 `filenames`")
        if not np.all([fname.endswith("_flt.fits") for fname in filenames]):
            raise ValueError(
                "`filenames` must all be `_flt.fits` files. e.g. {}".format(
                    filenames[0]
                )
            )
        if len(filenames) <= 12:
            return None
            raise ValueError("Not enough frames in visit.")
        with fits.open(
            filenames[0], cache=False, memmap=False, lazy_load_hdus=True
        ) as hdulist:
            wcs = WCS(hdulist[1])

        hdrs = [fits.getheader(file) for idx, file in enumerate(filenames)]
        exptime = np.asarray([hdr["EXPTIME"] for hdr in hdrs])
        filenames = filenames[exptime == np.median(exptime)]
        medexp = np.median(exptime)
        hdrs = [hdr for exp, hdr in zip(exptime, hdrs) if exp == medexp]

        if len(np.unique(np.asarray([hdr["TARGNAME"] for hdr in hdrs]))) != 1:
            raise ValueError("More than one target in files.")
        if len(np.unique(np.asarray([hdr["PROPOSID"] for hdr in hdrs]))) != 1:
            raise ValueError("More than one proposal id in files.")
        filters = np.asarray([hdr["FILTER"] for hdr in hdrs])
        mask = (filters == "G141") | (filters == "G102")

        convert_time = lambda hdr: (
            Time(
                datetime.strptime(
                    "{}-{}".format(hdr["DATE-OBS"], hdr["TIME-OBS"]),
                    "%Y-%m-%d-%H:%M:%S",
                )
            ).jd,
            Time(datetime.strptime("{}".format(hdr["DATE-OBS"]), "%Y-%m-%d")).jd,
        )
        time, start_date = np.asarray([convert_time(hdr) for hdr in hdrs]).T
        # maybe this is better
        time = np.asarray(
            [
                np.mean(
                    [
                        Time(hdr["EXPSTART"], format="mjd").jd,
                        Time(hdr["EXPEND"], format="mjd").jd,
                    ]
                )
                for hdr in hdrs
            ]
        )

        postarg1 = np.asarray([hdr["POSTARG1"] for hdr in hdrs])
        postarg2 = np.asarray([hdr["POSTARG2"] for hdr in hdrs])
        scan_length = np.median(np.asarray([hdr["SCAN_LEN"] for hdr in hdrs]))
        if scan_length > 0:
            if forward:
                mask &= postarg2 > 0
            else:
                mask &= postarg2 <= 0
            hdrs = [hdr for m, hdr in zip(mask, hdrs) if m]
            if not np.any(mask):
                return None

        sci, err, dq = [], [], []
        qmask = 1 | 2 | 4 | 8 | 16 | 32 | 256
        time = time[mask]
        for jdx, file in enumerate(filenames[mask]):
            with fits.open(file, cache=False, memmap=False) as hdulist:
                sci.append(hdulist[1].data)
                err.append(hdulist[2].data)
                dq.append(hdulist[3].data)
                if hdulist[1].header["BUNIT"] == "ELECTRONS":
                    sci[jdx] /= hdulist[1].header["SAMPTIME"]
                    err[jdx] /= hdulist[1].header["SAMPTIME"]
                err[jdx][(dq[jdx] & qmask) != 0] = 1e10

        if pixel_mask is not None:
            sci = [s * pixel_mask for s in sci]

        bright_pix = np.asarray([(sci1 > 100).sum() for sci1 in sci])
        if not force:
            if (bright_pix < 1000).any():
                return None
                raise ValueError("Not enough flux on target")

        X, Y = np.mgrid[: sci[0].shape[0], : sci[0].shape[1]]
        xs, ys = np.asarray(
            [
                [np.average(X[s >= 5], weights=s[s >= 5]) for s in sci],
                [np.average(Y[s >= 5], weights=s[s >= 5]) for s in sci],
            ]
        )

        arclength = np.hypot(xs - xs.min(), ys - ys.min())
        arclength -= np.median(arclength)
        if (np.abs(arclength) > 5).any():
            if not force:
                return None
                raise ValueError("Lost fine point")
            else:
                arclength_mask = ~sigma_clip(arclength).mask
                time = time[arclength_mask]
                sci = list(np.asarray(sci)[arclength_mask])
                err = list(np.asarray(err)[arclength_mask])
                dq = list(np.asarray(dq)[arclength_mask])

        if not force:
            if (
                np.nanmean(sci, axis=(0, 2)) > np.nanmean(sci, axis=(0, 2)).max() * 0.9
            ).sum() < 4:
                return None
                raise ValueError("Stare mode")

        s = np.argsort(time)
        return Visit(
            sci=np.asarray(sci)[s],
            err=np.asarray(err)[s],
            dq=np.asarray(dq)[s],
            time=time[s],
            exptime=hdrs[0]["EXPTIME"],
            name=hdrs[0]["TARGNAME"],
            forward=forward,
            propid=hdrs[0]["PROPOSID"],
            filenames=filenames,
            filter=hdrs[0]["FILTER"],
            visit_number=visit_number,
            ra=hdrs[0]["RA_TARG"],
            dec=hdrs[0]["DEC_TARG"],
            wcs=wcs,
            scan_length=scan_length,
        )

    @property
    def average_vsr(self):
        return self._slanted_vsr

    @property
    def average_spectrum(self):
        avg = np.atleast_3d(self.average_vsr)
        spec_mean = np.average(self.data / avg, weights=avg / self.error, axis=(0, 1))
        spec_mean /= np.mean(spec_mean)
        return (
            np.atleast_3d(spec_mean) * np.ones(self.shape).transpose([0, 2, 1])
        ).transpose([0, 2, 1])

    @property
    def _basic_vsr(self):
        avg = np.atleast_3d(
            np.average(self.data, weights=1 / self.error, axis=1)
        ).transpose([0, 2, 1])
        vsr_mean = np.asarray(
            [
                np.average(
                    self.data[idx] / avg[idx],
                    weights=avg[idx] / self.error[idx],
                    axis=1,
                )
                for idx in range(self.nt)
            ]
        )
        vsr_mean = np.atleast_3d(vsr_mean) * np.ones(self.shape)
        vsr_mean = (vsr_mean.T / np.mean(vsr_mean, axis=(1, 2))).T
        return vsr_mean

    @property
    def _basic_spectrum(self):
        avg = np.atleast_3d(self._basic_vsr)
        spec_mean = np.average(self.data / avg, weights=avg / self.error, axis=(0, 1))
        spec_mean /= np.mean(spec_mean)
        return (
            np.atleast_3d(spec_mean) * np.ones(self.shape).transpose([0, 2, 1])
        ).transpose([0, 2, 1])

    @property
    def average_lc(self):
        lc = np.average(
            (self.data / self.model), weights=(self.model / self.error), axis=(1, 2)
        )
        return lc

    @property
    def average_lc_err(self):
        return (
            np.average(
                ((self.data / self.model) - self.average_lc[:, None, None]) ** 2,
                weights=(self.model / self.error),
                axis=(1, 2),
            )
            ** 0.5
            / (np.product(self.shape[1:]) ** 0.5)
        )

    @property
    def basic_residuals(self):
        return self.data - (self.average_lc[:, None, None] * self.model)

    @property
    def residuals(self):
        if not hasattr(self, "full_model"):
            raise ValueError("Please run `fit_model`")
        return self.data - self.full_model

    def meta(self, pdx):
        meta = {
            attr: [
                getattr(self, attr)[pdx]
                if isinstance(getattr(self, attr), np.ndarray)
                else np.atleast_1d(getattr(self, attr))[0]
            ][0]
            for attr in [
                "ra",
                "dec",
                "period",
                "t0",
                "duration",
                "radius",
                "mass",
                "incl",
                "st_rad",
                "st_mass",
                "st_teff",
                "propid",
                "dist",
            ]
        }
        meta["tstart"] = self.time[0]
        meta["tend"] = self.time[-1]
        meta["ntime"] = len(self.time)
        return meta

    def get_flatfield(self):
        """ Get flat field of the data by fitting components in HST calibration files"""
        g141_flatfile = CALIPATH + "WFC3.IR.G141.flat.2.fits"
        ff_hdu = fits.open(g141_flatfile)

        nt, nrow, ncol = self.shape

        norm = np.nansum(self.data, axis=(1, 2))
        norm /= np.median(norm)
        norm = np.atleast_3d(norm).transpose([1, 0, 2])

        f = np.asarray([ff_hdu[i].data for i in range(4)])
        f = f[
            :,
            1014 // 2 - self.ns // 2 : 1014 // 2 + self.ns // 2,
            1014 // 2 - self.ns // 2 : 1014 // 2 + self.ns // 2,
        ]

        f = np.vstack([f, np.ones((1, f.shape[1], f.shape[2]))])
        X = f[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)][
            :, np.ones((nrow, ncol), bool)
        ].T
        avg = np.nanmean((self.data / norm / self.model), axis=0)[
            np.ones((nrow, ncol), bool)
        ]
        avg_err = (
            (1 / nt) * np.nansum(((self.error / norm / self.model)) ** 2, axis=0) ** 0.5
        )[np.ones((nrow, ncol), bool)]

        sigma_w_inv = np.dot(X.T, X / avg_err[:, None] ** 2) + np.diag(
            1 / (np.ones(5) * 1e10)
        )
        B = (
            np.dot(X.T, (avg / avg_err ** 2)[:, None])
            + np.asarray([1, 0, 0, 0, 0]) / 1e10
        )
        w = np.linalg.solve(sigma_w_inv, B)[:, 0]

        X = f[:, np.ones((self.ns, self.ns), bool)].T
        flat = np.dot(X, w).reshape((self.ns, self.ns))
        flat /= np.median(flat)
        flat[(self.dq[0] & 256) != 0] = 1
        flat = np.atleast_3d(flat).transpose([2, 0, 1])
        return flat

    def _prepare_design_matrix(self):
        """Make a design matrix for transit fitting"""
        grad = np.gradient(self.average_lc / self.average_lc.mean())[:8]
        hook_model = -np.exp(
            np.polyval(
                np.polyfit(
                    self.time[:8][grad > 0] - self.time[0],
                    np.log(grad[grad > 0]),
                    1,
                ),
                self.time - self.time[0],
            )
        )

        poly = np.vstack(
            [(self.time - self.time.mean()) ** idx for idx in np.arange(2)[::-1]]
        )

        A = np.vstack(
            [
                hook_model,
                self.xshift,
                np.gradient(self.xshift, self.time),
                self.yshift,
                np.gradient(self.yshift, self.time),
                self.xshift * self.yshift,
                self.bkg,
                poly,
            ]
        ).T

        def shift_check(time, xshift, yshift):
            """Find points where the telescope shifted during an observation."""
            dt = np.median(np.diff(time))
            k = np.where(np.diff(time) / dt > 3)[0] + 1
            if len(k) == 0:
                return None
            t, x, y = np.asarray(
                [
                    (t[0], x.mean(), y.mean())
                    for t, x, y in zip(
                        np.array_split(time, k),
                        np.array_split(xshift, k),
                        np.array_split(yshift, k),
                    )
                ]
            ).T
            v = np.hypot(np.diff(x) / np.diff(t), np.diff(y) / np.diff(t))
            mask = sigma_clip(v, sigma=3).mask
            if not mask.any():
                return None
            if mask.sum() > 1:
                mask = np.in1d(v, np.max(v[mask]))
            return np.where(np.in1d(time, t[1:][mask]))[0][0]

        break_point = shift_check(self.time, self.xshift, self.yshift)
        if break_point is not None:
            A2 = np.zeros((A.shape[0], A.shape[1] * 2 - 1))
            A2[:break_point, 1 : A.shape[1]] = A[:break_point, 1:]
            A2[break_point:, A.shape[1] : (A.shape[1]) * 2 - 1] = A[break_point:, 1:]
            A2[:, :1] = A[:, :1]
            return A2
        return A

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        **kwargs,
    ) -> plt.Axes:
        """Create a plot of the `Observation`.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes object, optional
            Optional axes object to plot into
        kwargs : dict
            Optional dictionary of keyword arguments to pass to
            matplotlib.pyplot.plot

        Returns
        -------
        ax :  matplotlib.pyplot.axes object, optional
             Plot of the `Spectrum`.
        """
        with plt.style.context("seaborn-white"):
            if ax is None:
                _, ax = plt.subplots()
            norm = np.median(self.average_lc[self.oot]) * np.ones(self.nt)
            ax.set(xlabel="Time [JD]", ylabel="$e^-s^{-1}$")
            if hasattr(self, "transits"):
                y = self.average_lc / self.noise_model
                plt.scatter(
                    self.time,
                    y / np.median(y[self.oot]),
                    c="k",
                    s=1,
                    label="Corrected",
                )
                if self.transits.sum() != 0:
                    plt.scatter(
                        self.time,
                        self.transits.sum(axis=-1) + 1,
                        s=2,
                        label="Transit",
                        c="C0",
                    )
                if self.eclipses.sum() != 0:
                    plt.scatter(
                        self.time,
                        self.eclipses.sum(axis=-1) + 1,
                        s=2,
                        label="Eclipse",
                        c="C1",
                    )
            else:
                ax.scatter(
                    self.time,
                    self.average_lc / norm,
                    s=1,
                    c="r",
                    label="Raw",
                )
            ax.legend()
            ax.set(xlim=xlim, ylim=ylim)
        return ax

    @property
    def oot(self):
        if not hasattr(self, "transits"):
            return np.ones(self.nt, bool)
        return ((self.eclipses + self.transits) == 0).all(axis=-1)

    def diagnose(self, frame: int = 0) -> plt.figure:
        """Make a diagnostic plot for the visit

        Parameters
        ----------
        frame: int
            Reference frame number

        Returns:
        fig: matplotlib.pyplot.figure
            Figure object
        """
        if not hasattr(self, "trace_wavelength"):
            raise ValueError("Calibrate the visit first with `calibrate` class method.")
        with plt.style.context("seaborn-white"):
            fig = plt.gcf()
            fig.set_size_inches(11, 5.5)
            plt.suptitle(self.name)
            for ax, y in zip(
                [plt.subplot2grid((2, 4), (0, 0)), plt.subplot2grid((2, 4), (0, 1))],
                [self.sci[frame], self.data[frame]],
            ):
                im = ax.imshow(
                    y,
                    vmin=0,
                    vmax=np.nanpercentile(y, 90),
                    cmap="viridis",
                )
                ax.set_aspect("auto")
                ax.set(
                    title=f"Frame {frame}",
                    xlabel="Spectral Pixel",
                    ylabel="Spatial Pixel",
                )
            plt.colorbar(im, ax=ax)

            ax = plt.subplot2grid((2, 4), (0, 2), colspan=2)
            ax.errorbar(
                self.time,
                self.average_lc / np.median(self.average_lc),
                self.average_lc_err / np.median(self.average_lc),
                c="k",
                markersize=0.4,
                marker=".",
                ls="",
            )
            if hasattr(self, "transits"):
                ax.scatter(
                    self.time,
                    self.noise_model,
                    c="purple",
                    label="Noise Model",
                )
                if np.nansum(self.transits) != 0:
                    ax.plot(
                        self.time,
                        self.transits.sum(axis=-1) + 1,
                        c="blue",
                        label="Transit model",
                    )
                if np.nansum(self.eclipses) != 0:
                    ax.plot(
                        self.time,
                        self.eclipses.sum(axis=-1) + 1,
                        c="red",
                        label="Eclipse model",
                    )
                ax.legend()
            ax.set(
                title="Average Light Curve",
                xlabel="Time",
                ylabel="Normalized Flux",
            )

            ax = plt.subplot2grid((2, 4), (1, 1))
            im = ax.pcolormesh(
                self.average_vsr[frame]
                * self.average_spectrum[frame]
                * self.average_lc[frame],
                cmap="viridis",
            )
            plt.colorbar(im, ax=ax)
            ax.set(
                title="Model",
                xlabel="Spectral Pixel",
                ylabel="Spatial Pixel",
                #                aspect="auto",
            )

            ax = plt.subplot2grid((2, 4), (1, 0))
            ax.plot(
                self.trace_wavelength,
                self.sensitivity_t / self.sensitivity_t.mean(),
                label="Model",
            )
            ax.plot(
                self.trace_wavelength,
                self.trace[~self.trace_mask] / self.trace[~self.trace_mask].mean(),
                label="Data",
            )
            ax.set(xlabel="Wavelength [A]", ylabel="Flux")
            ax.legend()

            ax = plt.subplot2grid((2, 4), (1, 2), colspan=2)
            im = ax.scatter(self.time, self.xshift, label="xshift", s=1)
            im = ax.scatter(self.time, self.yshift, label="yshift", s=1)
            im = ax.scatter(self.time, self.bkg, label="bkg", s=1)
            ax.legend()
            ax.set(
                title="Regressors",
                xlabel="Time",
                ylabel="Value",
                #                aspect="auto",
            )
            return fig

    def plot_residual_panel(self):
        fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
        plt.suptitle(self.__repr__())
        vlevel = np.max(
            np.abs(
                np.percentile(
                    self.basic_residuals,
                    [10, 90],
                )
            )
        )
        l = np.argmax(np.abs(np.diff(self.average_spectrum[0, 0, :]))[20:-20]) + 19
        for idx, ldx in enumerate(np.arange(l, l + 4)):
            for jdx, r in enumerate([self.basic_residuals, self.residuals]):
                ax = axs[jdx, idx]
                im = ax.imshow(
                    r[:, :, ldx].T,
                    cmap="coolwarm",
                    vmin=-vlevel,
                    vmax=vlevel,
                )

                if jdx == 0:
                    ax.set_title(f"Channel {ldx}")
                else:
                    ax.set_xlabel(f"Cadence Number")
                if idx == 0:
                    ax.set(ylabel="Spatial Pixel")

                ax.set_aspect("auto")
        cbar = plt.colorbar(im, ax=axs)
        cbar.set_label("Residual [e$^-$/s]")
        return fig

    @property
    def _slanted_vsr(self):
        k = np.abs(self.X) < -self.X[0][0][5]
        weights = np.copy(1 / self.error)
        weights[~k] = 0

        m = self.average_lc[:, None, None] * self._basic_spectrum
        frames = self.data / m
        frames_err = self.error / m
        avg = np.average(frames, weights=1 / frames_err, axis=(0))
        frames_err = frames_err / avg
        frames = frames / avg

        X = np.vstack(
            [self.X[0][0] ** 0, self.X[0][0], self.X[0][0] ** 2, self.X[0][0] ** 3]
        ).T
        model = np.zeros_like(self.data)
        prior_sigma = np.ones(4) * 10
        prior_mu = np.asarray([1, 0, 0, 0])

        for tdx in range(self.nt):
            for idx in range(self.nsp):
                sigma_w_inv = X[k[tdx][idx]].T.dot(
                    X[k[tdx][idx]] / frames_err[tdx][idx][k[tdx][idx], None] ** 2
                )
                sigma_w_inv += np.diag(1 / prior_sigma ** 2)
                B = X[k[tdx][idx]].T.dot(
                    frames[tdx][idx][k[tdx][idx]]
                    / frames_err[tdx][idx][k[tdx][idx]] ** 2
                )
                B += prior_mu / prior_sigma ** 2
                model[tdx][idx] = X.dot(np.linalg.solve(sigma_w_inv, B))

        vsr_mean = np.mean(model, axis=2)[:, :, None] * np.ones(self.shape)
        vsr_grad = model - vsr_mean
        return (vsr_mean + vsr_grad) * avg

    def build_noise_matrix(self):
        """Builds a sparse matrix containing all of the (wavelength dependent) transit components.

        The matrix will be npixels x (nwavelength bins * 2 + ld_npoly). It contains
        1) the transit model (without limb darkening) per each wavelength bin
        2) the eclipse model per each wavelength bin
        3) The difference between the limb darkened transit and un-limb darkened
            transit as a polynomial of order ld_npoly

        Parameters
        ----------
        ld_poly: int
            Polynomial order for limb darkening fit. If > 1, will allow a polynomial
            fit for delta limb darkening in wavelength
        """
        cdx = 0

        xs, ys, bkg = (
            self.xshift[:, None] * np.ones((self.nt, self.nsp)),
            self.yshift[:, None] * np.ones((self.nt, self.nsp)),
            self.bkg[:, None] * np.ones((self.nt, self.nsp)),
        )
        Y = self.Y[:, :, cdx]
        # X = (
        #     (np.arange(self.nt)[:, None, None] * np.ones(self.shape) - self.nt / 2)
        #     / (self.nt)
        # )[:, :, cdx]
        T = self.T[:, :, cdx]

        A = np.vstack(
            [
                np.asarray(
                    [T ** idx * Y ** jdx for idx in range(2) for jdx in range(2)]
                ),
                np.asarray(
                    [xs ** idx * ys ** jdx for idx in range(2) for jdx in range(2)][1:]
                ),
                np.asarray([xs * T ** idx for idx in np.arange(1, 2)]),
            ]
        )
        A1 = np.hstack(A.transpose([1, 0, 2])).T
        noise = vstack_independent(A1, self.nwav)
        # This reshapes so that we have neighboring channels next to each other in the design matrix
        As = sparse.hstack([noise[:, idx :: A1.shape[1]] for idx in range(A1.shape[1])])

        Anames = np.asarray(
            [
                *[f"$t^{idx}y^{jdx}$" for idx in range(2) for jdx in range(2)],
                #                *[f"$T^{idx}$" for idx in np.arange(1, 2)],
                *[f"$x_s^{idx}y_s^{jdx}$" for idx in range(2) for jdx in range(2)][1:],
                *[f"$x_s*t^{1}$"],
            ]
        )

        return Anames, As

    def build_transit_matrix(self, ld_npoly: int = 1, build_eclipse=True):
        """Builds a sparse matrix containing all of the (wavelength dependent) transit components.

        The matrix will be npixels x (nwavelength bins * 2 + ld_npoly). It contains
        1) the transit model (without limb darkening) per each wavelength bin
        2) the eclipse model per each wavelength bin
        3) The difference between the limb darkened transit and un-limb darkened
            transit as a polynomial of order ld_npoly

        Parameters
        ----------
        ld_poly: int
            Polynomial order for limb darkening fit. If > 1, will allow a polynomial
            fit for delta limb darkening in wavelength

        Returns
        -------
        A_names: list of str
            The names of each component in the matrix
        A: scipy.sparse matrix
            The design matrix
        """
        no_ld_t = self.no_limb_transits_subtime
        ld = self.transits_subtime - no_ld_t
        transit1 = vstack_independent(
            no_ld_t[np.ones(no_ld_t.shape[:2], bool), :], self.nwav
        )
        transit1 = sparse.hstack(
            [transit1[:, idx :: self.nplanets] for idx in range(self.nplanets)]
        )

        A_ld = [
            vstack(
                ld[np.ones(ld.shape[:2], bool), :].T,
                self.nwav,
                n_dependence=np.linspace(-0.5, 0.5, self.nwav) ** (idx + 1),
            )
            for idx in range(ld_npoly)
        ]
        if build_eclipse:
            eclipse = self.eclipses_subtime
            eclipse1 = vstack_independent(
                eclipse[np.ones(eclipse.shape[:2], bool), :], self.nwav
            )
            eclipse1 = sparse.hstack(
                [eclipse1[:, idx :: self.nplanets] for idx in range(self.nplanets)]
            )
            As = sparse.hstack([transit1, eclipse1, *A_ld], format="csr")
            Anames = np.hstack(
                [
                    [f"$\\delta f_{{tr, {letter}}}$" for letter in self.letter],
                    [f"$\\delta f_{{ec, {letter}}}$" for letter in self.letter],
                ]
            )
        else:
            As = sparse.hstack([transit1, *A_ld], format="csr")
            Anames = np.hstack(
                [
                    [f"$\\delta f_{{tr, {letter}}}$" for letter in self.letter],
                ]
            )

        return Anames, As

    def fit_model(self, suffix: str = "", ld_npoly: int = 1, build_eclipse=True):
        """
        Fits the eclipse/transit models for a given visit.

        Parameters
        ----------
        suffix: str
            Optional string to label the extracted transmission spectrum
        ld_poly: int
            Polynomial order for limb darkening fit. If > 1, will allow a polynomial
            fit for delta limb darkening in wavelength
        """

        if not hasattr(self, "As_noise"):
            self.Anames_noise, self.As_noise = self.build_noise_matrix()
        Anames_transit, As_transit = self.build_transit_matrix(
            ld_npoly=ld_npoly, build_eclipse=build_eclipse
        )
        Anames, As = (np.hstack([Anames_transit, self.Anames_noise])), sparse.hstack(
            [As_transit, self.As_noise]
        )

        avg = self.average_lc / self.average_lc.mean()
        y = ((self.data / self.model) / avg[:, None, None]).transpose([2, 0, 1]).ravel()
        yerr = (
            ((self.error / self.model) / avg[:, None, None])
            .transpose([2, 0, 1])
            .ravel()
        )

        prior_sigma = np.ones(As.shape[1]) * 1000000
        prior_inv = np.diag(1 / prior_sigma ** 2)

        oot = (
            self.transits_subtime.sum(axis=-1).sum(axis=1)
            + self.eclipses_subtime.sum(axis=-1).sum(axis=1)
        ) == 0
        k = np.ones(self.shape, bool).transpose([2, 0, 1]).ravel()
        for count in [0, 1]:
            # This makes the mask points have >5 sigma residuals
            c = (~k).astype(float) * 1e5 + 1
            sigma_w_inv = As.T.dot(As.multiply(1 / (yerr * c)[:, None] ** 2)).toarray()
            sigma_w_inv += prior_inv
            B = As.T.dot((y - y.mean()) / (yerr * c) ** 2)
            sigma_w = np.linalg.inv(sigma_w_inv)
            w = np.linalg.solve(sigma_w_inv, B)
            werr = sigma_w.diagonal() ** 0.5
            k &= np.abs(((y - y.mean()) - As.dot(w)) / yerr) < 5
        tds = np.abs(self.no_limb_transits_subtime.min(axis=(0, 1)))
        eds = np.abs(self.eclipses_subtime.min(axis=(0, 1)))
        oot_flux = np.median(self.average_lc[oot])

        # Package up result:
        if not hasattr(self, "transmission_spectrum"):
            self.transmission_spectrum, self.emission_spectrum = {}, {}

        for pdx in range(self.nplanets):
            td, ed, letter, meta = (
                tds[pdx],
                eds[pdx],
                self.letter[pdx],
                self.meta(pdx),
            )

            self.transmission_spectrum[
                f"{letter}_{suffix}" if suffix != "" else f"{letter}"
            ] = Spectrum(
                wavelength=self.wavelength.to(u.micron),
                spec=1e6
                * td
                * w[pdx * self.nwav : (pdx + 1) * self.nwav]
                / oot_flux
                * ppm,
                spec_err=1e6
                * td
                * werr[pdx * self.nwav : (pdx + 1) * self.nwav]
                / oot_flux
                * ppm,
                depth=td,
                name=self.name + f"{letter} Transmission Spectrum",
                visit=self.visit_number,
                meta=meta,
            )
            pdx += self.nplanets
            self.emission_spectrum[
                f"{letter}_{suffix}" if suffix != "" else f"{letter}"
            ] = Spectrum(
                wavelength=self.wavelength.to(u.micron),
                spec=1e6
                * ed
                * w[pdx * self.nwav : (pdx + 1) * self.nwav]
                / oot_flux
                * ppm,
                spec_err=1e6
                * ed
                * werr[pdx * self.nwav : (pdx + 1) * self.nwav]
                / oot_flux
                * ppm,
                depth=ed,
                name=self.name + f"{letter} Emission Spectrum",
                visit=self.visit_number,
                meta=meta,
            )

        self.full_model = (
            (
                (As.dot(w)).reshape((self.nwav, self.nt, self.nsp)).transpose([1, 2, 0])
                + y.mean()
            )
            * avg[:, None, None]
            * self.model
        )
        return w, sigma_w

    def to_fits(self):
        ivar = 1 / self.error ** 2
        var = 1 / ivar.sum(axis=1)
        err = np.sqrt(var)
        avg = var * (ivar * self.data).sum(axis=1)

        def table(x):
            hdu = fits.table_to_hdu(
                Table(
                    [
                        self.time,
                        avg[:, x] * u.count,
                        err[:, x] * u.count,
                    ],
                    names=["time", "flux", "flux_err"],
                )
            )
            hdu.name = f"{np.round(self.wavelength[x].value/1e4, 4)}"
            hdu.header["WVLENGTH"] = self.wavelength[x].value / 1e4
            return hdu

        hdu0 = fits.PrimaryHDU()
        hdr = hdu0.header
        hdr["Visit"] = self.visit_number
        hdr["Name"] = self.name
        for key, item in self.meta(0).items():
            if ~np.isfinite(item):
                continue
            hdr[key] = item
        for idx, letter in enumerate(self.letter):
            hdr[f"{letter}_period"] = self.period[idx]
            hdr[f"{letter}_t0"] = self.t0[idx]
            hdr[f"{letter}_dur"] = self.duration[idx] / 24
        hdulist = [hdu0, *[table(idx) for idx in range(self.nwav)]]
        hdulist = fits.HDUList(hdulist)
        return hdulist
