"""Deal with multiple visits"""
import warnings
from typing import List, Optional

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pymc3_ext as pmx
from astropy.io import fits
from dataclasses import dataclass
from tqdm import tqdm

from .matrix import vstack_list
from .query import download_target, get_nexsci
from .spec import Spectra, Spectrum
from .transit import fit_multi_transit
from .visit import Visit


@dataclass
class Observation(
    object,
):
    """Class for storing multiple visits"""

    visits: List[Visit]
    name: str
    period: [Optional] = None
    t0: [Optional] = None
    duration: [Optional] = None
    radius: [Optional] = None
    mass: [Optional] = None
    incl: [Optional] = None
    st_rad: [Optional] = None
    st_mass: [Optional] = None
    st_teff: [Optional] = None
    planets: [Optional] = None

    def __post_init__(self):
        (
            period,
            t0,
            duration,
            radius,
            mass,
            incl,
            letter,
            st_rad,
            st_mass,
            st_teff,
            dist,
        ) = get_nexsci(self, planets=self.planets)
        self.dist = dist
        (
            self.period,
            self.t0,
            self.duration,
            self.radius,
            self.mass,
            self.incl,
            self.letter,
            self.st_rad,
            self.st_mass,
            self.st_teff,
        ) = (
            self.period if self.period is not None else period,
            self.t0 if self.t0 is not None else t0,
            self.duration if self.duration is not None else duration,
            self.radius if self.radius is not None else radius,
            self.mass if self.mass is not None else mass,
            self.incl if self.incl is not None else incl,
            self.letter if self.incl is not None else letter,
            self.st_rad if self.st_rad is not None else st_rad,
            self.st_mass if self.st_mass is not None else st_mass,
            self.st_teff if self.st_teff is not None else st_teff,
        )
        self.incl = np.nan_to_num(self.incl.astype(float), nan=90)
        if self.planets is None:
            self.planets = self.letter
        self.nplanets = len(self.planets)
        for idx in range(len(self)):
            (
                self[idx].period,
                self[idx].t0,
                self[idx].duration,
                self[idx].radius,
                self[idx].mass,
                self[idx].incl,
                self[idx].letter,
                self[idx].st_rad,
                self[idx].st_mass,
                self[idx].st_teff,
                self[idx].dist,
                self[idx].nplanets,
                self[idx].planets,
            ) = (
                self.period,
                self.t0,
                self.duration,
                self.radius,
                self.mass,
                self.incl,
                self.letter,
                self.st_rad,
                self.st_mass,
                self.st_teff,
                self.dist,
                self.nplanets,
                self.planets,
            )
        for visit in self:
            visit.calibrate()
        return

    def __repr__(self):
        return "{} [{} planets, {} Visits]".format(self.name, self.nplanets, len(self))

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, s: int):
        if len(self.visits) == 0:
            raise ValueError("Empty Observation")
        return self.visits[s]

    @staticmethod
    def from_files(
        fnames: List[str], pixel_mask=None, limit=None, force=False, **kwargs
    ):
        """Make an Observation from files"""
        fnames = np.sort(fnames)
        times, postarg2 = [], []
        for f in fnames:
            with fits.open(f, lazy_load_hdus=True) as hdu:
                times.append(hdu[0].header["expstart"])
                postarg2.append(hdu[0].header["POSTARG2"])
        times, postarg2 = np.asarray([times, postarg2])
        s = np.argsort(times)
        times, postarg2, fnames = times[s], postarg2[s], fnames[s]
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
        for idx, mask in tqdm(
            enumerate(masks),
            total=len(masks),
            position=0,
            leave=True,
        ):
            for direction in [True, False]:
                if limit is not None:
                    if len(visits) >= limit:
                        break
                visit = Visit.from_files(
                    fnames[mask],
                    forward=direction,
                    visit_number=idx + 1,
                    force=force,
                    pixel_mask=pixel_mask,
                )
                if visit is not None:
                    visits.append(visit)
        if (not force) & (len(visits) == 0):
            raise ValueError("Can not extract visits, try `force`.")
        return Observation(visits, name=visits[0].name, **kwargs)

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

    def phase(self, idx=0):
        phase = ((self.time - self.t0[idx]) / self.period[idx]) % 1
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

    def meta(self, pdx):
        attrs = [
            "ra",
            "dec",
            "st_rad",
            "st_mass",
            "st_teff",
            "dist",
        ]
        if pdx != "star":
            attrs = np.hstack(
                [attrs, ["period", "t0", "duration", "radius", "mass", "incl"]]
            )

        dict = {
            attr: [
                getattr(self, attr)[pdx]
                if isinstance(getattr(self, attr), np.ndarray)
                else np.atleast_1d(getattr(self, attr))[0]
            ][0]
            for attr in attrs
        }
        return dict

    def plot_transit_fit(self, xlim=None, ylim=None, **kwargs) -> plt.Axes:
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
        fig :  matplotlib.pyplot.figure object
            Plot of the transit
        """
        with plt.style.context("seaborn-white"):
            npanels = 1
            if hasattr(self, "_transit_fit_inputs"):
                if self._transit_fit_inputs["x_suppl"] is not None:
                    npanels = 2
            fig, axs = plt.subplots(
                self.nplanets,
                npanels,
                figsize=(12, self.nplanets * 5),
                sharey=True,
                sharex=True,
            )
            if self.nplanets == 1:
                axs = np.atleast_2d(axs)
            norm = np.hstack(
                [
                    np.median(visit.average_lc[visit.oot]) * np.ones(visit.nt)
                    for visit in self
                ]
            )
            for pdx in range(self.nplanets):
                ax = axs[pdx, 0]
                ax.set(xlabel="Phase", ylabel="Normalized Flux")
                ax.scatter(
                    self.phase(pdx),
                    self.average_lc / norm,
                    s=1,
                    c="r",
                    label="Raw",
                )

                if hasattr(self, "noise_model"):
                    y = self.average_lc / self.noise_model
                    ax.scatter(
                        self.phase(pdx),
                        y / np.median(y[self.oot]),
                        c="k",
                        s=1,
                        label="Corrected",
                    )
                    if self.transits.sum() != 0:
                        ax.scatter(
                            self.phase(pdx),
                            self.transits.sum(axis=-1) + 1,
                            s=2,
                            label="Transit Model",
                            c="C0",
                        )

                    if self.eclipses.sum() != 0:
                        ax.scatter(
                            self.phase(pdx),
                            self.eclipses.sum(axis=-1) + 1,
                            s=2,
                            label="Eclipse Model",
                            c="C1",
                        )
                axs[pdx, 0].set(title=f"{self.name} {self.letter[pdx]} WFC3 Data")
                ax.legend(frameon=True)
                if npanels == 2:
                    phase_suppl = (
                        (self._transit_fit_inputs["x_suppl"] - self.t0[pdx])
                        / self.period[pdx]
                    ) % 1
                    phase_suppl[phase_suppl > 0.5] -= 1
                    axs[pdx, 1].scatter(
                        phase_suppl, self._transit_fit_inputs["y_suppl"], s=0.1, c="k"
                    )
                    axs[pdx, 1].scatter(
                        phase_suppl, self.transit_suppl + 1, s=1, c="C0"
                    )
                    if self.eclipses.sum() != 0:
                        axs[pdx, 1].scatter(
                            phase_suppl, self.eclipse_suppl + 1, s=1, c="C1"
                        )
                    axs[pdx, 1].set(
                        title=f"{self.name} {self.letter[pdx]} Supplementary Data"
                    )
                ax.set(xlim=xlim, ylim=ylim)
        return fig

    @property
    def _subtime_idxs(self):
        return np.hstack(
            [visit._subtime.ravel() ** 0 * idx for idx, visit in enumerate(self)]
        ).astype(int)

    @property
    def _subtime(self):
        return np.hstack([visit._subtime for visit in self])

    def _cast_subtime(self, name, loc=None):
        """Take either the map_soln or the trace and reshape the subtime arrays

        Parameters
        ----------
        name: str
            Name of the parameter to cast
        loc: None or int
            If none, will cast the map_soln. If int, will cast that index of the trace.
        """
        for visit in self:
            interp = []
            for pdx in range(self.nplanets):
                if loc == None:
                    y = self.map_soln[name][:, pdx]
                else:
                    y = np.vstack(self.trace.posterior[name])[loc][:, pdx]
                interp.append(
                    np.interp(
                        np.vstack(
                            [
                                np.linspace(
                                    visit.time[idx]
                                    - (visit.exptime * u.second.to(u.day)) / 2,
                                    visit.time[idx]
                                    + (visit.exptime * u.second.to(u.day)) / 2,
                                    visit.nsp,
                                )
                                for idx in range(visit.nt)
                            ]
                        ),
                        self._subtime,
                        y,
                    )
                )
            interp = np.asarray(interp).transpose([1, 2, 0])
            setattr(visit, name, interp)

    def _build_pymc3_model(self, point=None, fit=True, sample=True, draws=200):
        """We do this because we don't want to sample over the large transit/eclipse/subtime arrays
        They take up too much memory to have 1000s of samples of them.
        So instead we evaluate these outside of the model
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pymc3_model, map_soln, trace, self._pymc3_fit = fit_multi_transit(
                **self._transit_fit_inputs,
                fit=fit,
                point=point,
                sample=sample,
                draws=draws,
            )
        if fit:
            self._pymc3_model = pymc3_model
            self.map_soln = map_soln
        if sample:
            self.trace = trace

    def draw(self, loc=None):
        """Draw a samle from the trace"""

        if isinstance(loc, int):
            if not hasattr(self, "trace"):
                raise ValueError(
                    "can not draw with out a trace run `sample_transit` first"
                )
            nsamps = len(self.trace.sample_stats.draw) * len(
                self.trace.sample_stats.chain
            )
            if loc > nsamps:
                raise ValueError(f"Must specify an index less than {nsamp} samples")

        self._cast_subtime("no_limb_transits_subtime", loc)
        self._cast_subtime("transits_subtime", loc)
        if "eclipses" in self.map_soln:
            self._cast_subtime("eclipses_subtime", loc)
        else:
            for visit in self:
                visit.eclipses_subtime = np.zeros_like(visit.transits_subtime)

        if isinstance(loc, int):
            self.transits = np.vstack(self.trace.posterior["transits"])[loc]
            if "eclipses" in self.map_soln:
                self.eclipses = np.vstack(self.trace.posterior["eclipses"])[loc]
            self.noise_model = np.vstack(self.trace.posterior["noise_model"])[loc]
        else:
            self.transits = self.map_soln["transits"]
            self.noise_model = self.map_soln["noise_model"]

            if "transit_suppl" in self._pymc3_fit:
                self.transit_suppl = self._pymc3_fit["transit_suppl"]
            if "eclipses" in self.map_soln:
                self.eclipses = self.map_soln["eclipses"]
                if "transit_suppl" in self._pymc3_fit:
                    self.eclipse_suppl = self._pymc3_fit["eclipse_suppl"]
            else:
                self.eclipses = np.zeros_like(self.transits)
                if "transit_suppl" in self._pymc3_fit:
                    self.eclipse_suppl = np.zeros_like(self.transit_suppl)

        for idx in range(len(self)):
            self[idx].transits = self.transits[self._time_idxs == idx]
            self[idx].eclipses = self.eclipses[self._time_idxs == idx]
            self[idx].noise_model = self.noise_model[self._time_idxs == idx]
        return

    def fit_transit(
        self,
        x_suppl=None,
        y_suppl=None,
        yerr_suppl=None,
        fit_t0=True,
        fit_period=True,
        calc_eclipse=False,
        ttvs=False,
    ):

        self._transit_fit_inputs = {
            "x": self.time,
            "y": self.average_lc / np.median(self.average_lc),
            "yerr": self.average_lc_err / np.median(self.average_lc),
            "x_suppl": x_suppl,
            "y_suppl": y_suppl,
            "yerr_suppl": yerr_suppl,
            "r_val": self.radius,
            "t0_val": self.t0,
            "period_val": self.period,
            "inc_val": np.atleast_1d([np.deg2rad(i) for i in self.incl]),
            "r_star": self.st_rad,
            "m_star": self.st_mass,
            "exptime": np.median(
                [(visit.exptime * u.second.to(u.day)) for visit in self]
            ),
            "expsubtime": np.median(
                [(visit.exptime * u.second.to(u.day)) / visit.nsp for visit in self]
            ),
            "A": self.A,
            "offsets": self.offsets,
            "subtime": self._subtime,
            "letters": self.letter,
            "fit_t0": fit_t0,
            "fit_period": fit_period,
            "calc_eclipse": calc_eclipse,
            "ttvs": ttvs,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._build_pymc3_model(fit=True, sample=False)

        self.draw()
        if x_suppl is not None:
            k = (
                np.abs(
                    (
                        self._transit_fit_inputs["y_suppl"]
                        - self._pymc3_fit["transit_suppl"]
                        - self.map_soln["y_supplmean"]
                    )
                    / self._transit_fit_inputs["yerr_suppl"]
                )
                < 5
            )
            for attr in ["x_suppl", "y_suppl", "yerr_suppl"]:
                self._transit_fit_inputs[attr] = self._transit_fit_inputs[attr][k]
            #
            # k = (
            #     np.abs(
            #         (
            #             self._transit_fit_inputs["y"]
            #             - self.noise_model
            #             - self.transits.sum(axis=-1)
            #         )
            #         / self._transit_fit_inputs["yerr"]
            #     )
            #     < 10
            # )
            # self._transit_fit_inputs["yerr"][k] *= 10
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._build_pymc3_model(point=self.map_soln, fit=True, sample=False)
            self.draw()
        return

    def sample_transit(self, draws=5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._build_pymc3_model(
                point=self.map_soln, fit=False, sample=True, draws=draws
            )

    @property
    def oot(self):
        if hasattr(self, "transits"):
            return np.hstack(
                [
                    ((visit.transits == 0) & (visit.eclipses == 0)).all(axis=-1)
                    for visit in self
                ]
            )
        else:
            return np.hstack([np.ones(visit.nt, bool) for visit in self])

    def fit_model(self, spline=False, nsamps=50, ld_npoly=1):
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

        ndraws = (
            len(self.trace.sample_stats.draw) * len(self.trace.sample_stats.chain)
            if hasattr(self, "trace")
            else 0
        )
        for count in np.hstack(["map_soln", np.arange(ndraws)]):
            if count == "map_soln":
                # Make sure we're set to map_soln
                self.draw()
            else:
                # Draw from trace
                self.draw(int(count))
            [
                visit.fit_model(suffix=count, ld_npoly=ld_npoly)
                for visit in tqdm(
                    self,
                    desc=f"Fitting Spectra Per Visit [Draw {count}/{ndraws + 1}]",
                    total=len(self),
                    position=0,
                    leave=True,
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
                # spec = visit.average_spectrum[0, 0, :] / visit.sensitivity
                # c = np.mean(spec)
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
                        meta=self.meta("star"),
                    )
                )
        return Spectra(spectra, name=f"{self.name} Stellar Spectrum")

    def plot_ttvs(self):
        if f"tts_{self.letter[0]}" not in self._pymc3_model.test_point.keys():
            raise ValueError("Did not fit transit with TTVs enabled.")
        fig, ax = plt.subplots(
            self.nplanets,
            figsize=(6, 3 * self.nplanets),
            facecolor="white",
            sharex=True,
            sharey=True,
        )
        if self.nplanets == 1:
            ax = [ax]
        for idx, letter in enumerate(self.letter):
            orig_t0s = pmx.eval_in_model(
                getattr(self._pymc3_model, f"tts_{letter}"),
                model=self._pymc3_model,
                point=self._pymc3_model.test_point,
            )

            ax[idx].scatter(
                orig_t0s + self.t0[idx],
                orig_t0s - self.map_soln[f"tts_{letter}"],
                s=1,
                c="k",
            )
            if hasattr(self, "trace"):
                if self.trace is not None:
                    tts = np.vstack(
                        np.atleast_3d(self.trace.posterior[f"tts_{letter}"])
                    )
                    ax[idx].errorbar(
                        orig_t0s + self.t0[idx],
                        orig_t0s - tts.mean(axis=0),
                        tts.std(axis=0),
                        c="k",
                        ls="",
                    )
            ax[idx].set(
                ylabel="$\delta$ Transit Mid Point [days]",
                title=f"{self.name} {letter}",
            )
            if idx == self.nplanets - 1:
                ax[idx].set(xlabel="Original Transit Mid Point [JD]")
        return fig

    def plot_spectra(self, type="transmission"):
        fig = plt.figure(
            figsize=((self.nplanets + 1) * 6, 3 * len(self)), facecolor="white"
        )
        for idx, visit in enumerate(self):
            ax = plt.subplot2grid((len(self), self.nplanets + 1), (idx, 0))
            if idx != len(self) - 1:
                ax.set(xlabel="")
            visit.plot(ax=ax)
            for jdx, letter in enumerate(self.letter):
                ax = plt.subplot2grid((len(self), self.nplanets + 1), (idx, jdx + 1))
                getattr(visit, f"{type}_spectrum")[f"{letter}_map_soln"].plot(
                    ax=ax, c=f"C{[0 if type == 'transmission' else 1][0]}"
                )
                if idx != len(self) - 1:
                    ax.set(xlabel="")
                if jdx != 0:
                    ax.set(ylabel="")

        return fig
