import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits


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

    def write(self):
        raise NotImplementedError


class Spectrum(object):
    """Helper object to carry  spectra"""

    def __init__(self, wavelength, spec, spec_err, visit, depth, name=None):
        self.wavelength = wavelength
        self.spec = spec
        self.spec_err = spec_err
        self.visit = visit
        self.name = name

    @property
    def table(self):
        return [
            pd.DataFrame(
                np.vstack(
                    [
                        self.wavelength[idx],
                        self.spec[idx],
                        self.spec_err[idx],
                    ]
                ).T,
                columns=["wavelength", "spec", "spec_err"],
            )
            for idx in range(self.nv)
        ]

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
