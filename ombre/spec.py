import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.units import cds

cds.enable()


class Spectra(object):
    """Helper object to carry multiple spectra"""

    def __init__(self, spec=[], name=None):
        if not isinstance(spec, list):
            raise ValueError("pass list")

        self.spec = spec
        self.name = name

    @property
    def visits(self):
        return [t.visit for t in self.spec]

    def append(self, obj):
        self.spec.append(obj)

    def __getitem__(self, s):
        if len(self.spec) == 0:
            raise ValueError("Empty Spectra")
        return self.spec[s]

    def __repr__(self):
        if len(self.spec) == 0:
            return "Empty Spectra"
        return f"Spectra [Visits: {self.visits}]"

    def plot(self, ax=None, **kwargs):
        with plt.style.context("seaborn-white"):
            if ax is None:
                _, ax = plt.subplots()
            for ts in self:
                ts.plot(ax=ax, label=f"Visit {ts.visit}", **kwargs)
            ax.legend()
            ax.set_title(f"{self.name}")

    @property
    def hdulist(self):
        if len(self.spec) == 0:
            raise ValueError("Empty Spectra")
        hdr = fits.Header()
        hdr["ORIGIN"] = "ombre"
        hdr["TARGET"] = f"{self.name}"
        hdr["DATE"] = Time.now().isot.split("T")[0]
        hdr["AUTHOR"] = "Christina Hedges (christina.l.hedges@nasa.gov)"
        hdr["VISITS"] = ", ".join(np.asarray(self.visits, str))

        phdu = fits.PrimaryHDU(header=hdr)
        hdulist = [phdu]

        for spec in self:
            hdulist.append(spec.hdu)
        return fits.HDUList(hdulist)


class Spectrum(object):
    """Helper object to carry a spectrum"""

    def __init__(self, wavelength, spec, spec_err, visit, depth, meta=None, name=None):
        self.wavelength = wavelength
        self.spec = spec
        self.spec_err = spec_err
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

    def __getitem__(self, s):
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
    def table(self):
        return Table(
            [
                (spec.wavelength / 1e4) * u.micron,
                spec.spec * cds.ppm,
                spec.spec_err * cds.ppm,
            ],
            names=["wavelength", "spectrum", "spectrum_err"],
        )

    def __repr__(self):
        return f"Spectrum [Visits {self.visit}]"

    def plot(self, ax=None, **kwargs):
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
