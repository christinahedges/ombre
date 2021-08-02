"""Deal with multiple visits"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from .visit import Visit
from .query import get_nexsci
from .matrix import vstack_list
from .modeling import fit_transit


@dataclass
class Observation(object):
    """Class for storing multiple visits"""

    visits: List[Visit]
    name: str

    def __post_init__(self):
        get_nexsci(self)
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
    def from_files(fnames: List[str]):
        """Make an Observation from files"""
        unique = np.unique([fname.split("/")[-1][:3] for fname in fnames])
        masks = np.asarray(
            [fname.split("/")[-1][:3] == unq for unq in unique for fname in fnames]
        ).reshape((len(unique), len(fnames)))

        visits = [
            Visit.from_files(fnames[mask], forward=direction)
            for mask in masks
            for direction in [True, False]
        ]
        return Observation(visits, name=visits[0].name)

    def fit_transit(self):
        """Fit the transits in all the visits"""
        return

    def _cast_transit_to_visits(self):
        """Puts transit data into the visits"""
        return

    @property
    def time(self):
        return np.hstack([visit.time for visit in self])

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
        return ((self.time - self.t0) / self.period - 0.5) % 1

    @property
    def A(self):
        return vstack_list([visit.A for visit in self])

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
            ax.scatter(self.phase, self.average_lc, s=1)
            ax.set(xlabel="Phase", ylabel="$e^-s^{-1}$")
        return ax

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
        )
