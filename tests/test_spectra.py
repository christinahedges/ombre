import os

import numpy as np
import pytest

import ombre as om
from ombre import PACKAGEDIR, __version__


def is_action():
    try:
        return os.environ["GITHUB_ACTIONS"]
    except KeyError:
        return False


def test_spectra():
    spectrum1 = om.Spectrum(2, 1, 0.1, visit=1, name="Christina")
    spectrum2 = om.Spectrum(
        np.arange(10), np.ones(10), np.ones(10) * 0.01, visit=2, name="Christina"
    )
    spectra = om.Spectra([spectrum1, spectrum2], name="Christina")
    assert len(spectrum1.table) == 1
    assert len(spectrum2.table) == 10
    # Can't make a spectrum with the wrong number of elements
    with pytest.raises(ValueError):
        spectrum1 = om.Spectrum(2, [1, 1], 0.1, visit=1, name="Christina")
    # Check reprs
    spectrum1
    spectra
    # Can't pass just one spectrum
    with pytest.raises(ValueError):
        spectra = om.Spectra(spectrum1)
    # Can't pass not a spectrum
    with pytest.raises(ValueError):
        spectra = om.Spectra("Christina")
