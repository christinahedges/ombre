import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.units import cds

cds.enable()


class Spectra(object):
    """Helper object to carry  spectra"""

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

    def to_hdu(self):
        hdr = fits.Header()
        hdr["ORIGIN"] = "ombre"
        hdr["TARGET"] = f"{self.name}"
        hdr["DATE"] = Time.now().isot.split("T")[0]
        hdr["DEPTH"] = self.depth
        hdr["AUTHOR"] = "Christina Hedges (christina.l.hedges@nasa.gov)"
        hdr["VISITS"] = ", ".join(np.asarray(self.visits, str))

        phdu = fits.PrimaryHDU(header=hdr)
        hdulist = [phdu]

        for spec in self:
            visits = spec.visit
            if not hasattr(visits, "__iter__"):
                visits = [visits]

            tab = Table(
                [
                    (spec.wavelength / 1e4) * u.micron,
                    spec.spec * cds.ppm,
                    spec.spec_err * cds.ppm,
                ],
                names=["wavelength", "spectrum", "spectrum_err"],
            )

            hdu = fits.table_to_hdu(tab)
            hdr = hdu.header
            hdr["Visit"] = ", ".join(np.asarray(visits, str))
            hdr["Name"] = spec.name

            if spec.meta is not None:
                for key, item in meta.items():
                    hdr[key] = item

            hdulist.append(hdu)

        return fits.HDUList(hdulist)


class Spectrum(object):
    """Helper object to carry  spectra"""

    def __init__(self, wavelength, spec, spec_err, visit, depth, meta=None, name=None):
        self.wavelength = wavelength
        self.spec = spec
        self.spec_err = spec_err
        self.visit = visit
        self.name = name
        self.depth = depth
        self.meta = meta

    @property
    def table(self):
        return pd.DataFrame(
            np.vstack(
                [
                    self.wavelength,
                    self.spec,
                    self.spec_err,
                ]
            ).T,
            columns=["wavelength", "spec", "spec_err"],
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
