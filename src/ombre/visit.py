"""Class for handling visit data"""
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Tuple, Union, Optional, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
from glob import glob
from datetime import datetime
import warnings

from astropy.io import fits, votable
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.convolution import convolve, Box2DKernel
from . import PACKAGEDIR
from .modeling import fit_transit
from .calibrate import wavelength_calibrate
from .matrix import fit_model

from urllib.request import URLError

from .query import get_nexsci

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
    cadence_mask: Optional[npt.NDArray[bool]] = None
    name: Optional[str] = None
    forward: bool = True
    visit_number: Optional[int] = 1
    propid: Optional[int] = 0
    name: Optional[str] = None
    filter: Optional[str] = "G141"
    filenames: Optional[List[str]] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    #    t0: Optional[float] = None
    #    period: Optional[float] = None

    def __post_init__(self):
        """Runs after init"""
        # self.sys = from_nexsci(self.name, limb_darkening=[0, 0])
        # if self.t0 is not None:
        #     self.sys.secondaries[0].t0 = self.t0
        # else:
        #     self.t0 = self.sys.secondaries[0].t0.eval()
        # if self.period is not None:
        #     self.sys.secondaries[0].period = self.period
        # else:
        #     self.period = self.sys.secondaries[0].porb.eval()
        self.ns = self.sci.shape[1]
        if self.ns == 1024:
            self.ns = 1014
        self.mask, self.spectral, self.spatial = simple_mask(
            self.sci, filter=self.filter
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
        self.model = (
            self.average_spectrum * self.average_vsr  # * self.average_lc[:, None, None]
        )

        self._build_regressors()
        self._subtime = dts = np.vstack(
            [
                np.linspace(
                    self.time[idx] - (self.exptime * u.second.to(u.day)) / 2,
                    self.time[idx] + (self.exptime * u.second.to(u.day)) / 2,
                    self.nsp,
                )
                for idx in range(self.nt)
            ]
        )
        self.A = self._prepare_design_matrix()

    def __repr__(self):
        return "{} Visit {}, Forward {} [Proposal ID: {}]".format(
            self.name, self.visit_number, self.forward, self.propid
        )

    def calibrate(self):
        if not hasattr(self, "st_teff"):
            get_nexsci(self)
        self.wavelength, self.sensitivity, self.sensitivity_t = wavelength_calibrate(
            self
        )

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
            "wavelength",
            "sensitivity",
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

        bkg = sigma_clipped_stats(
            np.ma.masked_array(
                self.sci, larger_mask[None, :, :] * np.ones(self.sci.shape, bool)
            ),
            axis=(1, 2),
            sigma=3,
        )[1]
        self.bkg = bkg - np.median(bkg)

    @staticmethod
    def from_files(
        filenames=List[str],
        visit_number: int = 1,
        forward: bool = True,
        t0: Optional[float] = None,
        period: Optional[float] = None,
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
            raise ValueError("Not enough frames in visit.")
        hdrs = [fits.getheader(file) for idx, file in enumerate(filenames)]
        exptime = np.asarray([hdr["EXPTIME"] for hdr in hdrs])
        filenames = filenames[exptime == np.median(exptime)]
        hdrs = [fits.getheader(file) for idx, file in enumerate(filenames)]

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
        if forward:
            mask = postarg2 > 0
        else:
            mask = postarg2 <= 0
        if not np.any(mask):
            raise ValueError("No files exist with that direction")

        sci, err, dq = [], [], []
        qmask = 1 | 2 | 4 | 8 | 16 | 32 | 256

        for jdx, file in enumerate(filenames[mask]):
            with fits.open(file, cache=False, memmap=False) as hdulist:
                sci.append(hdulist[1].data)
                err.append(hdulist[2].data)
                dq.append(hdulist[3].data)
                if hdulist[1].header["BUNIT"] == "ELECTRONS":
                    sci[jdx] /= hdulist[1].header["SAMPTIME"]
                    err[jdx] /= hdulist[1].header["SAMPTIME"]
                err[jdx][(dq[jdx] & qmask) != 0] = 1e10

        for sci1 in sci:
            if (sci1 > 100).sum() < 2000:
                raise ValueError("Not enough flux on target")

        X, Y = np.mgrid[: sci[0].shape[0], : sci[0].shape[1]]
        xs, ys = np.asarray(
            [
                [np.average(X, weights=s * (s > 100)) for s in sci],
                [np.average(Y, weights=s * (s > 100)) for s in sci],
            ]
        )
        arclength = np.hypot(xs - xs.min(), ys - ys.min())
        if (arclength > 5).any():
            raise ValueError("Lost fine point")

        if (
            np.nanmean(sci, axis=(0, 2)) > np.nanmean(sci, axis=(0, 2)).max() * 0.9
        ).sum() < 4:
            raise ValueError("Stare mode")
        s = np.argsort(time[mask])
        return Visit(
            sci=np.asarray(sci)[s],
            err=np.asarray(err)[s],
            dq=np.asarray(dq)[s],
            time=time[mask][s],
            exptime=hdrs[0]["EXPTIME"],
            name=hdrs[0]["TARGNAME"],
            forward=forward,
            propid=hdrs[0]["PROPOSID"],
            filenames=filenames[mask][s],
            filter=hdrs[0]["FILTER"],
            visit_number=visit_number,
            ra=hdrs[0]["RA_TARG"],
            dec=hdrs[0]["DEC_TARG"],
            #            t0=t0,
            #            period=period,
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
    def residuals(self):
        return self.data - (self.average_lc[:, None, None] * self.model)

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

        sigma_w_inv = np.dot(X.T, X / avg_err[:, None] ** 2)
        B = np.dot(X.T, (avg / avg_err ** 2)[:, None])
        w = np.linalg.solve(sigma_w_inv, B)[:, 0]

        X = f[:, np.ones((self.ns, self.ns), bool)].T
        flat = np.dot(X, w).reshape((self.ns, self.ns))
        flat /= np.median(flat)
        flat[(self.dq[0] & 256) != 0] = 1
        flat = np.atleast_3d(flat).transpose([2, 0, 1])
        return flat

    def _prepare_design_matrix(self):
        """Make a design matrix for transit fitting"""
        if not hasattr(self, "st_teff"):
            get_nexsci(self)
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
            [(self.time - self.time.mean()) ** idx for idx in np.arange(3)[::-1]]
        )

        return np.vstack(
            [
                self.xshift,
                self.yshift,
                self.xshift * self.yshift,
                self.bkg,
                hook_model,
                poly,
            ]
        ).T

    def fit_transit(self):
        self._pymc3_model, self.map_soln = fit_transit(
            x=self.time,
            y=self.average_lc / np.median(self.average_lc),
            yerr=self.average_lc_err / np.median(self.average_lc),
            r_val=self.radius,
            t0_val=self.t0,
            period_val=self.period,
            inc_val=np.deg2rad(self.incl),
            r_star=self.st_rad,
            m_star=self.st_mass,
            exptime=np.median(np.diff(self.time)),
            A=self.A,
            offsets=self.A[:, -1][:, None],
            subtime=self._subtime,
        )
        self.no_limb_transit_subtime = self.map_soln["no_limb_transit_subtime"].reshape(
            (self.nt, self.nsp)
        )
        self.transit_subtime = self.map_soln["transit_subtime"].reshape(
            (self.nt, self.nsp)
        )
        self.eclipse_subtime = self.map_soln["eclipse_subtime"].reshape(
            (self.nt, self.nsp)
        )

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
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
            ax.scatter(
                self.time,
                self.average_lc / norm,
                s=1,
                c="r",
                label="Raw",
            )
            ax.set(xlabel="Time [JD]", ylabel="$e^-s^{-1}$")
            if hasattr(self, "transit"):
                y = self.average_lc / self.noise_model
                plt.scatter(
                    self.time,
                    y / np.median(y[self.oot]),
                    c="k",
                    s=1,
                    label="Corrected",
                )
                if self.transit.sum() != 0:
                    plt.scatter(self.time, self.transit + 1, s=2, label="Transit")
                if self.eclipse.sum() != 0:
                    plt.scatter(self.time, self.eclipse + 1, s=2, label="Eclipse")
            ax.legend()
        return ax

    @property
    def oot(self):
        if not hasattr(self, "transit"):
            return np.ones(self.nt, bool)
        return (self.eclipse + self.transit) == 0

    def fit_model(self, spline=False, nknots=30, nsamps=40):
        """
        Fits the eclipse/transit models for a given visit.

        Parameters
        ----------

        spline: bool
            Whether to use a spline model for the transit depth
            If True, will use splines. This will make the spectrum
            "smooth"
        nknots: int
            Number of knots for the spline
        nsamps: int
            Number of samples to draw for each spectrum
        """
        fit_model(self, spline=spline, nknots=nknots, nsamps=nsamps)

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
        with plt.style.context("seaborn-white"):
            fig = plt.gcf()
            fig.set_size_inches(11, 8.5)
            plt.suptitle(self.name)
            for ax, y in zip(
                [plt.subplot2grid((3, 4), (0, 0)), plt.subplot2grid((3, 4), (0, 1))],
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
            # ax = plt.subplot2grid((3, 4), (0, 2))
            # ax.plot(
            #     self.average_vsr[frame],
            #     c="k",
            #     lw=0.1,
            # )
            # ax.set(
            #     title="Average Spatial Scan",
            #     xlabel="Spatial Pixel",
            #     ylabel="Normalized Flux",
            # )
            #
            # ax = plt.subplot2grid((3, 4), (0, 3))
            # ax.plot(
            #     self.average_spectrum[frame].T,
            #     c="k",
            #     lw=0.5,
            # )
            # ax.set(
            #     title="Average Spectrum",
            #     xlabel="Spectral Pixel",
            #     ylabel="Normalized Flux",
            # )
            #
            # ax = plt.subplot2grid((3, 4), (0, 2))
            # ax.imshow(
            #     self.flat[0],
            #     cmap="coolwarm",
            #     vmin=0.99,
            #     vmax=1.01,
            # )
            # ax.set(
            #     title="Flat Field",
            #     xlabel="Spectral Pixel",
            #     ylabel="Spatial Pixel",
            #     #                aspect="auto",
            # )

            ax = plt.subplot2grid((3, 4), (0, 2), colspan=2)
            ax.errorbar(
                self.time,
                self.average_lc / np.median(self.average_lc),
                self.average_lc_err / np.median(self.average_lc),
                c="k",
                markersize=0.4,
                marker=".",
                ls="",
            )
            if hasattr(self, "transit"):
                # ax.scatter(
                #     self.time,
                #     self.map_soln["full_model"],
                #     c="r",
                #     s=0.1,
                #     label="Data",
                # )
                # ax.plot(
                #     self.time,
                #     self.map_soln["hook"] + 1,
                #     c="lime",
                # )
                ax.scatter(
                    self.time,
                    self.noise_model,
                    c="purple",
                    label="Noise Model",
                )
                if np.nansum(self.transit) != 0:
                    ax.plot(
                        self.time,
                        self.transit + 1,
                        c="blue",
                        label="Transit model",
                    )
                if np.nansum(self.eclipse) != 0:
                    ax.plot(
                        self.time,
                        self.eclipse + 1,
                        c="red",
                        label="Eclipse model",
                    )
                ax.legend()
            ax.set(
                title="Average Light Curve",
                xlabel="Time",
                ylabel="Normalized Flux",
            )

            # ax = plt.subplot2grid((3, 4), (2, 0))
            # im = ax.imshow(
            #     self.average_spectrum[frame],
            #     cmap="viridis",
            # )
            # plt.colorbar(im, ax=ax)
            # ax.set(
            #     title="Average Spectrum",
            #     xlabel="Spectral Pixel",
            #     ylabel="Spatial Pixel",
            #     aspect="auto",
            # )
            # ax = plt.subplot2grid((3, 4), (2, 1))
            # im = ax.imshow(
            #     self.average_vsr[frame],
            #     cmap="viridis",
            # )
            # plt.colorbar(im, ax=ax)
            # ax.set(
            #     title="Average VSR",
            #     xlabel="Spectral Pixel",
            #     ylabel="Spatial Pixel",
            #     aspect="auto",
            # )

            ax = plt.subplot2grid((3, 4), (1, 1))
            im = ax.imshow(
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

            vlevel = np.max(
                np.abs(
                    np.percentile(
                        self.residuals,
                        [10, 90],
                    )
                )
            )
            ax = plt.subplot2grid((3, 4), (1, 0))
            # im = ax.imshow(
            #     self.residuals[frame],
            #     cmap="coolwarm",
            #     vmin=-vlevel,
            #     vmax=vlevel,
            # )
            # plt.colorbar(im, ax=ax)
            # ax.set(
            #     title="Residuals",
            #     xlabel="Spectral Pixel",
            #     ylabel="Spatial Pixel",
            #     #                aspect="auto",
            # )

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

            ax = plt.subplot2grid((3, 4), (1, 2), colspan=2)
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
            l = np.argmax(np.abs(np.diff(self.average_spectrum[0, 0, :]))[20:-20]) + 19

            for idx, ldx in enumerate(np.arange(l, l + 4)):
                ax = plt.subplot2grid((3, 4), (2, idx))
                im = ax.imshow(
                    self.residuals[:, :, ldx].T,
                    cmap="coolwarm",
                    vmin=-vlevel,
                    vmax=vlevel,
                )
                plt.colorbar(im, ax=ax)
                ax.set(
                    title=f"Residuals Channel [{ldx}]",
                    xlabel="Time",
                    ylabel="Spatial Pixel",
                    #                    aspect="auto",
                )

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


def simple_mask(
    sci: npt.NDArray[np.float64], filter: str = "G141"
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = sci / mask
        data[~np.isfinite(data)] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    len_secs = np.diff(np.where(np.gradient(secs) != 0)[0][::2] + 1)[::2]
    if len(len_secs) > 1:
        mask[:, :, : np.where(np.gradient(secs) > 0)[0][3]] &= False
        spectral = mask[0].any(axis=0)

    return (
        mask,
        np.atleast_3d(np.atleast_2d(spectral)).transpose([2, 0, 1]),
        np.atleast_3d(np.atleast_2d(spatial.T)).transpose([2, 1, 0]),
    )
