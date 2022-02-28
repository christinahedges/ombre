"""Tools for getting literature values from NExSci"""
import logging
import os
import sys
from functools import wraps
from urllib.request import URLError

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import G, k_B, m_p
from astropy.io import votable
from astropy.utils.data import conf, download_file
from astroquery.mast import Observations as astropyObs
from tqdm import tqdm

log = logging.getLogger("ombre")


def suppress_stdout(f, *args):
    """A simple decorator to suppress function print outputs."""

    @wraps(f)
    def wrapper(*args):
        # redirect output to `null`
        with open(os.devnull, "w") as devnull:
            old_out = sys.stdout
            sys.stdout = devnull
            try:
                return f(*args)
            # restore to default
            finally:
                sys.stdout = old_out

    return wrapper


def get_nexsci(input, planets=None, **kwargs):
    if isinstance(input, (tuple, list, np.ndarray)):
        ra, dec = np.copy(input)
    else:
        ra, dec = np.copy(input.ra), np.copy(input.dec)
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+ra,dec,pl_orbper,pl_tranmid,pl_trandur,pl_rade,pl_bmassj,pl_orbincl,st_rad,st_raderr1,st_raderr2,st_mass,st_masserr1,st_masserr2,st_teff,sy_dist,pl_letter,hostname+from+pscomppars+where+"
        f"ra+>+{ra - 0.0083333}+and+ra+<+{ra + 0.0083333}+and+dec+>{dec - 0.0083333}+and+dec+<{dec + 0.0083333}"
    )
    try:
        with conf.set_temp("remote_timeout", 20):
            df = votable.parse(url).get_first_table().to_table().to_pandas()
        df = df.sort_values("pl_orbper")
        df = df.dropna(subset=["pl_orbper", "pl_tranmid"]).reset_index(drop=True)
        if planets is not None:
            df = df[
                np.any([np.in1d(df.pl_letter, letter) for letter in planets], axis=0)
            ]

        # df = df[df.pl_letter == letter].reset_index(drop=True)
        (ra, dec, period, t0, duration, radius, mass, incl,) = (
            np.asarray(df).T[:8].astype(float)
        )
        (st_rad, rer1, rer2, st_mass, mer1, mer2, st_teff, dist) = (
            np.asarray(df).T[8:-2, 0].astype(float)
        )
        pl_letter = np.asarray(df.pl_letter)
    except URLError:
        raise URLError(
            "Can not access the internet to query NExSci for exoplanet parameters."
        )
    except IndexError:
        return 0, 0, 0, 0, 0, 90, "b", (0, 0), (0, 0), 6000, 0

    radius *= u.earthRad.to(u.solRad)
    mass *= u.jupiterMass.to(u.solMass)
    rer = np.nanmax(np.abs([rer1, rer2]))
    mer = np.nanmax(np.abs([mer1, mer2]))
    return (
        period,
        t0,
        duration,
        radius,
        mass,
        incl,
        pl_letter,
        (st_rad, rer if rer != 0 else st_rad * 0.2),
        (st_mass, mer if mer != 0 else st_mass * 0.2),
        st_teff,
        dist,
    )


# @suppress_stdout
def download_target(targetname, radius="10 arcsec", download_dir=None):
    """Downloads the WFC3 observations for a target"""
    if download_dir is None:
        download_dir = os.path.join(os.path.expanduser("~"), ".ombre-cache")
    if not os.path.isdir(download_dir):
        try:
            os.mkdir(download_dir)
        # downloads locally if OS error occurs
        except OSError:
            log.warning(
                "Warning: unable to create {}. "
                "Downloading MAST files to the current "
                "working directory instead.".format(download_dir)
            )
            download_dir = "."

    logging.getLogger("astropy").setLevel(log.getEffectiveLevel())
    obsTable = astropyObs.query_criteria(
        target_name=targetname, obs_collection="HST", project="HST", radius=radius
    )
    obsTable = obsTable[
        (obsTable["instrument_name"] == "WFC3/IR")
        & (obsTable["dataRights"] == "PUBLIC")
    ]

    fnames = [
        "{}/mastDownload/".format(download_dir)
        + url.replace("mast:", "").replace("product", url.split("/")[-1].split("_")[0])[
            :-9
        ]
        + "_flt.fits"
        for url in obsTable["dataURL"]
    ]
    log.info("Found {} files.".format(len(obsTable)))
    paths = []
    for idx, t in enumerate(
        tqdm(obsTable, desc=f"Downloading files ({targetname}) ", total=len(obsTable))
    ):
        if os.path.isfile(fnames[idx]):
            paths.append(fnames[idx])
        else:
            t1 = astropyObs.get_product_list(t)
            t1 = t1[t1["productSubGroupDescription"] == "FLT"]
            paths.append(
                astropyObs.download_products(
                    t1, mrp_only=False, download_dir=download_dir
                )["Local Path"][0]
            )
    return paths


def get_nexsci_df(cache=True):
    NEXSCI_API = (
        "http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    )
    url1 = NEXSCI_API + (
        "?table=planets&select=pl_hostname,pl_letter,"
        "pl_disc,ra,dec,pl_trandep,pl_tranmid,pl_tranmiderr1,"
        "pl_tranmiderr2,pl_tranflag,pl_trandur,pl_pnum,pl_k2flag,"
        "pl_kepflag,pl_facility,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,"
        "pl_orblper,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,"
        "st_raderr2,st_teff,st_tefferr1,st_tefferr2,st_optmag,st_j,st_h"
    )
    url2 = NEXSCI_API + (
        "?table=compositepars&select=fpl_hostname,fpl_letter,"
        "fpl_smax,fpl_smaxerr1,fpl_smaxerr2,fpl_radj,fpl_radjerr1,"
        "fpl_radjerr2,fpl_bmassj,fpl_bmassjerr1,fpl_bmassjerr2,"
        "fpl_bmassprov,fpl_eqt,fpl_orbper,fpl_orbpererr1,fpl_orbpererr2,"
        "fpl_eccen,fpl_eccenerr1,fpl_eccenerr2,fst_spt"
    )

    path1 = download_file(url1, cache=cache, show_progress=False, pkgname="ombre")
    path2 = download_file(url2, cache=cache, show_progress=False, pkgname="ombre")

    planets = pd.read_csv(
        path1,
        comment="#",
        skiprows=1,
    )
    composite = pd.read_csv(
        path2,
        comment="#",
        skiprows=1,
    )
    composite.rename(
        {c: c[1:] for c in composite.columns if c.startswith("fpl")},
        axis="columns",
        inplace=True,
    )
    df = pd.merge(
        left=planets,
        right=composite,
        how="left",
        left_on=["pl_hostname", "pl_letter"],
        right_on=["pl_hostname", "pl_letter"],
    )
    df["st_teff"][df.pl_hostname == "GJ 436"] = 3684.0
    df["a"] = (
        (
            (
                G
                * (
                    np.asarray(df.st_mass) * u.solMass
                    + np.asarray(df.pl_bmassj) * u.jupiterMass
                )
                / 4
                * np.pi ** 2
            )
            * (np.asarray(df.pl_orbper) * u.day) ** 2
        )
        ** (1 / 3)
    ).to(u.solRad)
    df["Teq"] = (
        np.asarray(df.st_teff) * u.K * (np.asarray(df.st_rad) / (2 * df["a"])) ** 0.5
    )
    df["g"] = (
        (G * np.asarray(df.pl_bmassj) * u.jupiterMass)
        / (np.asarray(df.pl_radj) * u.jupiterRad) ** 2
    ).to(u.m / u.s ** 2)
    df["H"] = (
        k_B
        * np.asarray(df["Teq"])
        * u.K
        / (2 * m_p * np.asarray(df["g"]) * (u.m / u.s ** 2))
    ).to(u.km)
    df["cross_sec"] = (
        ((np.asarray(df["pl_radj"]) * u.jupiterRad) + 5 * np.asarray(df["H"]) * u.km)
        ** 2
        - ((np.asarray(df["pl_radj"]) * u.jupiterRad)) ** 2
    ).to(u.solRad ** 2)
    df["delta_tran_dep"] = (
        ((np.asarray(df["pl_radj"]) * u.jupiterRad) + 5 * np.asarray(df["H"]) * u.km)
        ** 2
        - ((np.asarray(df["pl_radj"]) * u.jupiterRad)) ** 2
    ).to(u.solRad ** 2) / (np.asarray(df["st_rad"]) * u.solRad) ** 2
    df["density"] = (
        np.asarray(df["pl_bmassj"], float)
        * u.jupiterMass
        / (4 / 3 * np.pi * (np.asarray(df["pl_radj"], float) * u.jupiterRad) ** 3)
    ).to(u.g / u.cm ** 3)
    return df
