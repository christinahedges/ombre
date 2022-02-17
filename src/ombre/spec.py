"""Classes for handling spectra"""
from typing import Generic, Optional, Tuple, TypeVar, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from lightkurve.units import ppm

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
        if not hasattr(self.wavelength, "unit"):
            self.wavelength *= u.micron
        if not hasattr(self.spec, "unit"):
            self.spec *= ppm
        if not hasattr(self.spec_err, "unit"):
            self.spec_err *= ppm
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

    def bin(self, bins: Optional[npt.NDArray] = None):
        if bins is None:
            bins = np.linspace(0.8, 1.7, 100) * u.micron
        if not hasattr(bins, "unit"):
            bins *= u.micron
        bins = bins.to(self.wavelength.unit)
        y, ye = (
            np.zeros(len(bins) - 1) * np.nan * self.spec.unit,
            np.zeros(len(bins) - 1) * np.nan * self.spec_err.unit,
        )
        for idx in range(len(bins) - 1):
            k = (
                (self.wavelength > bins[idx])
                & (self.wavelength <= bins[idx + 1])
                & (np.isfinite(self.spec))
            )
            if k.sum() == 0:
                continue
            if k.sum() > 4:
                if np.nansum(self.spec_err) == 0:
                    y[idx] = np.average(self.spec[k])
                    ye[idx] = np.average((self.spec[k] - y[idx]) ** 2) ** 0.5 / (
                        k.sum() ** 0.5
                    )
                else:
                    y[idx] = np.average(self.spec[k], weights=1 / self.spec_err[k])
                    ye[idx] = np.average(
                        (self.spec[k] - y[idx]) ** 2, weights=1 / self.spec_err[k]
                    ) ** 0.5 / (k.sum() ** 0.5)
            else:
                y[idx] = np.mean(self.spec[k])
                ye[idx] = np.sum(self.spec_err[k] ** 2) ** 0.5 / (k.sum())
        bins = bins[:-1] + np.median(np.diff(bins)) / 2
        return Spectrum(
            bins,
            y,
            ye,
            visit=self.visit,
            depth=self.depth,
            meta=self.meta,
            name=self.name,
        )

    @property
    def table(self) -> Table:
        """Spectrum as an astropy.table.Table object"""
        return Table(
            [
                (self.wavelength),
                self.spec,
                self.spec_err,
            ],
            names=["wavelength", "spectrum", "spectrum_err"],
        )

    def __len__(self):
        return len(self.spec)

    def __repr__(self):
        return f"Spectrum [Visit {self.visit}]"

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
            if np.nansum(self.spec_err) != 0:
                ax.errorbar(
                    self.wavelength.value,
                    self.spec.value,
                    self.spec_err.value,
                    **kwargs,
                )
            else:
                ax.plot(
                    self.wavelength.value,
                    self.spec.value,
                    **kwargs,
                )
            plt.title(f"{self.name}, Visit: {self.visit}")
            plt.xlabel("Wavelength")
            plt.ylabel("$\delta$ Transit Depth [ppm]")
        return ax

    @staticmethod
    def from_file(fname, ext=1):
        hdu = fits.open(fname)
        keys = np.asarray(list(hdu[ext].header.keys()))
        keys = keys[np.where(keys == "DEPTH")[0][0] + 1 :]
        meta = {key.lower(): hdu[ext].header[key] for key in keys}
        self = Spectrum(
            hdu[ext].data["wavelength"] * u.Unit(hdu[ext].header["TUNIT1"]),
            hdu[ext].data["spectrum"],
            hdu[ext].data["spectrum_err"],
            visit=hdu[ext].header["VISIT"],
            name=hdu[ext].header["NAME"],
            depth=hdu[ext].header["DEPTH"],
            meta=meta,
        )

        return self

    @property
    def hdu(self):
        """Spectrum as an astropy.io.fits.HDU object. Use this to write a fits file."""

        hdu = fits.table_to_hdu(self.table)
        hdr = hdu.header
        hdr["Visit"] = self.visit
        hdr["Name"] = self.name
        hdr["DEPTH"] = self.depth

        if self.meta is not None:
            for key, item in self.meta.items():
                if ~np.isfinite(item):
                    continue
                hdr[key] = item
        return hdu

    @property
    def hdulist(self):
        """Spectrum as an astropy.io.fits.HDU object. Use this to write a fits file."""

        hdr = fits.Header()
        hdr["ORIGIN"] = "ombre"
        hdr["TARGET"] = f"{self.name}"
        hdr["DATE"] = Time.now().isot.split("T")[0]
        hdr["AUTHOR"] = "Christina Hedges (christina.l.hedges@nasa.gov)"
        phdu = fits.PrimaryHDU(header=hdr)
        hdulist = [phdu]
        hdulist.append(self.hdu)
        return fits.HDUList(hdulist)


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
        if hasattr(s, "__iter__"):
            return Spectra([self.spec[s1] for s1 in s])
        return self.spec[s]

    def __repr__(self):
        if len(self.spec) == 0:
            return "Empty Spectra"
        return f"Spectra [Visits: {self.visits}]"

    def plot(
        self, ax: Optional[plt.Axes] = None, legend: bool = False, **kwargs
    ) -> plt.Axes:
        """Create a plot of the `Spectra`.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes object, optional
            Optional axes object to plot into
        legend: bool
            If True, will add a legend to the plot
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
            if legend:
                ax.legend()
            ax.set_title(f"{self.name}")
        return ax

    def flatten(self):
        return Spectrum(
            np.hstack([spec.wavelength for spec in self]),
            np.hstack([spec.spec for spec in self]),
            np.hstack([spec.spec_err for spec in self]),
            name=self[0].name,
            depth=self[0].depth,
            meta=self[0].meta,
            visit=0,
        )

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
        try:
            hdr["VISITS"] = ", ".join(
                np.asarray(
                    [
                        ", ".join(np.asarray(v, str)) if hasattr(v, "__iter__") else v
                        for v in self.visits
                    ],
                    str,
                )
            )
        except:
            pass
        phdu = fits.PrimaryHDU(header=hdr)
        hdulist = [phdu]
        for spec in self:
            hdulist.append(spec.hdu)
        return fits.HDUList(hdulist)

    @staticmethod
    def from_file(fname):
        with fits.open(fname) as hdu:
            specs = []
            for ext in np.arange(1, len(hdu)):
                keys = np.asarray(list(hdu[ext].header.keys()))
                keys = keys[np.where(keys == "DEPTH")[0][0] + 1 :]
                meta = {key.lower(): hdu[ext].header[key] for key in keys}
                specs.append(
                    Spectrum(
                        hdu[ext].data["wavelength"] * u.Unit(hdu[ext].header["TUNIT1"]),
                        hdu[ext].data["spectrum"],
                        hdu[ext].data["spectrum_err"],
                        visit=hdu[ext].header["VISIT"],
                        name=hdu[ext].header["NAME"],
                        depth=hdu[ext].header["DEPTH"],
                        meta=meta,
                    )
                )
        return Spectra(specs, specs[0].name)
