import os
from glob import glob

import numpy as np
import pytest

import ombre as om
from ombre import PACKAGEDIR, __version__


def is_action():
    try:
        return os.environ["GITHUB_ACTIONS"]
    except KeyError:
        return False


def test_version():
    assert __version__ == "0.2.0"


def test_visit():
    # v = om.Visit(
    #     sci=np.random.normal(100, 0.01, size=(2, 64, 64)),
    #     err=np.ones((2, 64, 64)) * 0.01,
    #     dq=np.zeros((2, 64, 64)),
    #     time=np.ones(2),
    #     exptime=0.02,
    #     name="Christina",
    # )
    # assert v.shape == (2, 64, 64)
    # # test repr
    # v
    fnames = glob("/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/data/hd209458b/*")
    v = om.Visit.from_files(fnames)
    # Would normally run get_nexsci but don't want to ping in the tests
    v.st_teff = 6000
    v.calibrate()
    v.diagnose(frame=0).savefig("test_hd209458b.pdf")
    # fnames = glob("/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/data/xo1b/*")
    # v = om.Visit.from_files(fnames)
    # assert v.shape == (127, 23, 128)
    # assert v.period == 3.94153
    # v.diagnose(frame=30).savefig("test_xo1b.pdf")
