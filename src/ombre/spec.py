"""Classes for handling spectra"""
import numpy as np
import numpy.typing as npt
from typing import Optional, Union
from typing import TypeVar, Generic, Tuple, Union, Optional
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.units import cds

cds.enable()

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Spectrum(object):
    def __init__(
        self,
        wavelength: npt.NDArray[Union[np.float64, int]],
        spec: npt.NDArray[Union[np.float64, int]],
        spec_err: npt.NDArray[Union[np.float64, int]],
        visit: int = 1,
        depth: Optional[float] = None,
        meta: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """Helper object to carry a single spectrum

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in microns
        spec : np.ndarray
            Spectrum array
        spec_err : np.ndarray
            Spectrum error array
        visit : int
            Visit number. Default is 1.
        depth : float, optional
            Transit depth associated with spectrum
        meta : dict, optional
            Dictionary of metadata
        name : str, optional
            Name of the target
        """
        self.wavelength = np.atleast_1d(wavelength)
        self.spec = np.atleast_1d(spec)
        self.spec_err = np.atleast_1d(spec_err)
        if not np.all(
            np.asarray([len(getattr(self, attr)) for attr in ["spec", "spec_err"]])
            == len(self.wavelength)
        ):
            raise ValueError(
                "`spec`, `spec_err`, and `wavelength` must all be the same length"
            )
        if hasattr(visit, "__iter__"):
            if len(visit) == 1:
                self.visit = visit[0]
            else:
                self.visit = visit
        else:
            self.visit = visit

        self.name = name
        self.depth = depth
        self.meta = meta

    def __getitem__(self, s: int):
        return Spectrum(
            self.wavelength[s],
            self.spec[s],
            self.spec_err[s],
            self.visit,
            self.depth,
            meta=self.meta,
            name=self.name,
        )

    @property
    def table(self) -> Table:
        """Spectrum as an astropy.table.Table object"""
        return Table(
            [
                (self.wavelength / 1e4) * u.micron,
                self.spec * cds.ppm,
                self.spec_err * cds.ppm,
            ],
            names=["wavelength", "spectrum", "spectrum_err"],
        )

    def __len__(self):
        return len(self.spec)

    def __repr__(self):
        return f"Spectrum [Visits {self.visit}]"

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Create a plot of the `Spectrum`.

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
            k = self.spec_err < 1e5
            ax.errorbar(
                self.wavelength,
                self.spec,
                self.spec_err,
                **kwargs,
            )
            plt.title(f"{self.name}, Visit: {self.visit}")
            plt.xlabel("Wavelength [A]")
            plt.ylabel("$\delta$ Transit Depth [ppm]")
        return ax

    @property
    def hdu(self):
        """Spectrum as an astropy.io.fits.HDU object. Use this to write a fits file."""
        visits = self.visit
        if not hasattr(visits, "__iter__"):
            visits = [visits]

        tab = self.table

        hdu = fits.table_to_hdu(tab)
        hdr = hdu.header
        hdr["Visit"] = ", ".join(np.asarray(visits, str))
        hdr["Name"] = self.name
        hdr["DEPTH"] = self.depth

        if self.meta is not None:
            for key, item in self.meta.items():
                hdr[key] = item
        return hdu


class Spectra(object):
    def __init__(self, spec: list = [], name: Optional[str] = None):
        """Helper object to carry multiple spectra

        Parameters
        ----------
        spec : list
            List of Spectrum objects.
        name : str, optional
            Name of the dataset
        """
        if not isinstance(spec, list):
            raise ValueError("Please pass a list of `Spectrum` objects")
        if not isinstance(spec[0], Spectrum):
            raise ValueError("Please pass a list of `Spectrum` objects")
        self.spec = spec
        self.name = name

    @property
    def visits(self) -> list:
        """List of visits in the Spectra object"""
        return [t.visit for t in self.spec]

    def append(self, obj: Spectrum) -> None:
        """Appends a new Spectrum to the Spectra object"""
        if isinstance(obj, Spectrum):
            self.spec.append(obj)
        else:
            raise ValueError("Can only append a `Spectrum` object")

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, s: int):
        if len(self.spec) == 0:
            raise ValueError("Empty Spectra")
        return self.spec[s]

    def __repr__(self):
        if len(self.spec) == 0:
            return "Empty Spectra"
        return f"Spectra [Visits: {self.visits}]"

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Create a plot of the `Spectra`.

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
             Plot of the `Spectra`.
        """
        with plt.style.context("seaborn-white"):
            if ax is None:
                _, ax = plt.subplots()
            for ts in self:
                ts.plot(ax=ax, label=f"Visit {ts.visit}", **kwargs)
            ax.legend()
            ax.set_title(f"{self.name}")

    @property
    def hdulist(self):
        """Spectra as an astropy.io.fits.HDUList object. Use this to write a fits file with the `write` method."""
        if len(self.spec) == 0:
            raise ValueError("Empty Spectra")
        hdr = fits.Header()
        hdr["ORIGIN"] = "ombre"
        hdr["TARGET"] = f"{self.name}"
        hdr["DATE"] = Time.now().isot.split("T")[0]
        hdr["AUTHOR"] = "Christina Hedges (christina.l.hedges@nasa.gov)"
        hdr["VISITS"] = ", ".join(
            np.asarray(
                [
                    ", ".join(np.asarray(v, str)) if hasattr(v, "__iter__") else v
                    for v in self.visits
                ],
                str,
            )
        )
        phdu = fits.PrimaryHDU(header=hdr)
        hdulist = [phdu]
        for spec in self:
            hdulist.append(spec.hdu)
        return fits.HDUList(hdulist)
