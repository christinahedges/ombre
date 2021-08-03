from astropy.io import votable
import astropy.units as u
from astroquery.mast import Observations as astropyObs
from urllib.request import URLError
from tqdm import tqdm
import os
import logging


log = logging.getLogger("ombre")


def get_nexsci(input, **kwargs):
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+ra,dec,pl_orbper,pl_tranmid,pl_trandur,pl_rade,pl_orbincl,st_rad,st_mass,st_teff+from+pscomppars+where+"
        f"ra+>+{input.ra - 0.0083333}+and+ra+<+{input.ra + 0.0083333}+and+dec+>{input.dec - 0.0083333}+and+dec+<{input.dec + 0.0083333}"
    )
    try:
        (ra, dec, period, t0, duration, radius, incl, st_rad, st_mass, st_teff,) = (
            votable.parse(url).get_first_table().to_table().to_pandas().iloc[0]
        )
    except URLError:
        raise URLError(
            "Can not access the internet to query NExSci for exoplanet parameters."
        )

    radius *= u.earthRad.to(u.solRad)
    return period, t0, duration, radius, incl, st_rad, st_mass, st_teff


def download_target(targetname, radius="10 arcsec"):
    """Downloads the WFC3 observations for a target"""
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
        tqdm(obsTable, desc="Downloading files", total=len(obsTable))
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
