"""Deal with multiple visits"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from .visit import Visit
from .query import get_nexsci, download_target
from .matrix import vstack_list
from .modeling import fit_transit
from .spec import Spectrum, Spectra
from astropy.io import fits
import warnings

from tqdm import tqdm


@dataclass
class Observation(
    object,
):
    """Class for storing multiple visits"""

    visits: List[Visit]
    name: str
    planet_letter: str = "b"
    period: [Optional] = None
    t0: [Optional] = None
    duration: [Optional] = None
    radius: [Optional] = None
    incl: [Optional] = None
    st_rad: [Optional] = None
    st_mass: [Optional] = None
    st_teff: [Optional] = None

    def __post_init__(self):
        period, t0, duration, radius, incl, st_rad, st_mass, st_teff = get_nexsci(
            self, letter=self.planet_letter
        )
        (
            self.period,
            self.t0,
            self.duration,
            self.radius,
            self.incl,
            self.st_rad,
            self.st_mass,
            self.st_teff,
        ) = (
            [self.period if self.period is not None else period][0],
            [self.t0 if self.t0 is not None else t0][0],
            [self.duration if self.duration is not None else duration][0],
            [self.radius if self.radius is not None else radius][0],
            [self.incl if self.incl is not None else incl][0],
            [self.st_rad if self.st_rad is not None else st_rad][0],
            [self.st_mass if self.st_mass is not None else st_mass][0],
            [self.st_teff if self.st_teff is not None else st_teff][0],
        )
        for idx in range(len(self)):
            (
                self[idx].period,
                self[idx].t0,
                self[idx].duration,
                self[idx].radius,
                self[idx].incl,
                self[idx].st_rad,
                self[idx].st_mass,
                self[idx].st_teff,
            ) = (
                self.period,
                self.t0,
                self.duration,
                self.radius,
                self.incl,
                self.st_rad,
                self.st_mass,
                self.st_teff,
            )
        for visit in self:
            visit.calibrate()
        return

    def __repr__(self):
        return "{} [{} Visits]".format(self.name, len(self))

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, s: int):
        if len(self.visits) == 0:
            raise ValueError("Empty Observation")
        return self.visits[s]

    @staticmethod
    def from_files(fnames: List[str], planet_letter="b", **kwargs):
        """Make an Observation from files"""
        fnames = np.sort(fnames)
        times = []
        for f in fnames:
            with fits.open(f, lazy_load_hdus=True) as hdu:
                times.append(hdu[0].header["expstart"])
        times = np.asarray(times)
        s = np.argsort(times)
        times, fnames = times[s], fnames[s]
        breaks = np.hstack(
            [0, np.where(np.diff(times) > (180 / (60 * 24)))[0] + 1, len(fnames)]
        )
        masks = [
            np.in1d(np.arange(len(fnames)), np.arange(b1, b2))
            for b1, b2 in zip(breaks[:-1], breaks[1:])
        ]
        # unique = np.unique([fname.split("/")[-1][:5] for fname in fnames])
        # masks = np.asarray(
        #     [fname.split("/")[-1][:5] == unq for unq in unique for fname in fnames]
        # ).reshape((len(unique), len(fnames)))

        visits = []
        for direction in [True, False]:
            for idx, mask in enumerate(masks):
                try:
                    visits.append(
                        Visit.from_files(
                            fnames[mask], forward=direction, visit_number=idx + 1
                        )
                    )
                except ValueError:
                    continue
        return Observation(
            visits, name=visits[0].name, planet_letter=planet_letter, **kwargs
        )

    @staticmethod
    def from_MAST(targetname, download_dir=None, radius="10 arcsec", **kwargs):
        """Download a target from MAST"""
        paths = np.asarray(
            download_target(targetname, radius=radius, download_dir=download_dir)
        )
        return Observation.from_files(paths, **kwargs)

    @property
    def time(self):
        return np.hstack([visit.time for visit in self])

    @property
    def _time_idxs(self):
        return np.hstack(
            [visit.time ** 0 * idx for idx, visit in enumerate(self)]
        ).astype(int)

    @property
    def average_lc(self):
        return np.hstack([visit.average_lc for visit in self])

    @property
    def average_lc_err(self):
        return np.hstack([visit.average_lc_err for visit in self])

    @property
    def ra(self):
        ra = [visit.ra for visit in self]
        if np.all(np.abs(np.diff(ra)) < 1e-1):
            return np.median(ra)
        else:
            raise ValueError("Visits do not seem to be of the same target")

    @property
    def dec(self):
        dec = [visit.dec for visit in self]
        if np.all(np.abs(np.diff(dec)) < 1e-1):
            return np.median(dec)
        else:
            raise ValueError("Visits do not seem to be of the same target")

    @property
    def phase(self):
        phase = ((self.time - self.t0) / self.period) % 1
        phase[phase > 0.5] -= 1
        return phase

    @property
    def A(self):
        return vstack_list([visit.A for visit in self])

    @property
    def offsets(self):
        return self.A[
            :, self.A.shape[1] // len(self) - 1 :: self.A.shape[1] // len(self)
        ]

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
            norm = np.hstack(
                [
                    np.median(visit.average_lc[visit.oot]) * np.ones(visit.nt)
                    for visit in self
                ]
            )
            ax.scatter(
                self.phase,
                self.average_lc / norm,
                s=1,
                c="r",
                label="Raw",
            )
            ax.set(xlabel="Phase", ylabel="$e^-s^{-1}$")
            if hasattr(self, "map_soln"):
                y = self.average_lc / self.map_soln["noise_model"]
                plt.scatter(
                    self.phase,
                    y / np.median(y[self.oot]),
                    c="k",
                    s=1,
                    label="Corrected",
                )
                if self.map_soln["transit"].sum() != 0:
                    plt.scatter(
                        self.phase, self.map_soln["transit"] + 1, s=2, label="Transit"
                    )
                if self.map_soln["eclipse"].sum() != 0:
                    plt.scatter(
                        self.phase, self.map_soln["eclipse"] + 1, s=2, label="Eclipse"
                    )
            ax.legend()
        return ax

    @property
    def _subtime_idxs(self):
        return np.hstack(
            [visit._subtime.ravel() ** 0 * idx for idx, visit in enumerate(self)]
        ).astype(int)

    @property
    def _subtime(self):
        return np.hstack([visit._subtime.ravel() for visit in self])

    def fit_transit(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                offsets=self.offsets,
                subtime=self._subtime,
                **kwargs,
            )
        self.period = self.map_soln["period"]
        self.t0 = self.map_soln["t0"]
        for idx in range(len(self)):
            self[idx].no_limb_transit_subtime = self.map_soln[
                "no_limb_transit_subtime"
            ][self._subtime_idxs == idx].reshape((self[idx].nt, self[idx].nsp))
            self[idx].transit_subtime = self.map_soln["transit_subtime"][
                self._subtime_idxs == idx
            ].reshape((self[idx].nt, self[idx].nsp))
            self[idx].eclipse_subtime = self.map_soln["eclipse_subtime"][
                self._subtime_idxs == idx
            ].reshape((self[idx].nt, self[idx].nsp))
            self[idx].transit = self.map_soln["transit"][self._time_idxs == idx]
            self[idx].eclipse = self.map_soln["eclipse"][self._time_idxs == idx]
            self[idx].noise_model = self.map_soln["noise_model"][self._time_idxs == idx]
            self[idx].period = self.period
            self[idx].t0 = self.t0

    @property
    def oot(self):
        if hasattr(self, "map_soln"):
            return np.hstack(
                [(visit.transit == 0) & (visit.eclipse == 0) for visit in self]
            )
        else:
            return np.hstack([np.ones(visit.nt, bool) for visit in self])

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
        [
            visit.fit_model(spline=spline, nknots=nknots, nsamps=nsamps)
            for visit in tqdm(
                self, desc="Fitting Transit/Eclipse Model", total=len(self)
            )
        ]

    @property
    def stellar_spectra(self):
        spectra = []
        for filter in ["G141", "G102"]:
            w, y, ye = [], [], []
            for visit in self:
                if visit.filter != filter:
                    continue
                spec = visit.average_spectrum[0, 0, :] / visit.sensitivity
                c = np.mean(spec)
                w.append(visit.wavelength)
                y.append(spec / c)
                ye.append(np.zeros(visit.nwav))
            if len(w) > 0:
                w, y, ye = np.hstack(w), np.hstack(y), np.hstack(ye)
                s = np.argsort(w)
                spectra.append(
                    Spectrum(
                        w[s],
                        y[s],
                        ye[s],
                        name=f"{self.name} Normalized Stellar Spectrum",
                        visit=filter,
                    )
                )
        return Spectra(spectra, name=f"{self.name} Stellar Spectrum")

    def plot_spectra(self):
        fig = plt.figure(figsize=(14, 5))
        with plt.style.context("seaborn-white"):
            plotted = False
            for visit in self:
                if not plotted:
                    ax = plt.subplot2grid((1, 2), (0, 0), fig=fig)
                    cdx = 0
                    plotted = True
                if np.nansum(visit.transmission_spectrum.spec) != 0:
                    ax = visit.transmission_spectrum_draws.plot(
                        alpha=0.1, ax=ax, color=f"C{cdx}"
                    )
                    #        ax.set_title(f'{visit.name} Transmission Spectrum Visit {visit.visit_number}')
                    cdx += 1

            plotted = False
            for visit in self:
                if np.nansum(visit.emission_spectrum.spec) != 0:
                    if not plotted:
                        ax = plt.subplot2grid((1, 2), (0, 1), fig=fig)
                        cdx = 0
                        plotted = True
                    ax = visit.emission_spectrum_draws.plot(
                        alpha=0.1, ax=ax, color=f"C{cdx}"
                    )
                    #            ax.set_title(f'{visit.name} Emission Spectrum Visit {visit.visit_number}')
                    cdx += 1
