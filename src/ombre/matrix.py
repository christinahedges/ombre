"""Matrix operations for ombre"""

import numpy as np
from scipy import sparse
from astropy.time import Time
import matplotlib.pyplot as plt
import astropy.units as u
import lightkurve as lk

from .spec import Spectrum, Spectra
from lightkurve.units import ppm


def vstack_list(dms):
    npoints = np.sum([dm.shape[0] for dm in dms])
    ncomps = np.sum([dm.shape[1] for dm in dms])
    if sparse.issparse(dms[0]):
        X = sparse.lil_matrix((npoints, ncomps))
    else:
        X = np.zeros((npoints, ncomps))
    idx = 0
    jdx = 0
    for dm in dms:
        X[idx : idx + dm.shape[0], jdx : jdx + dm.shape[1]] += dm
        idx = idx + dm.shape[0]
        jdx = jdx + dm.shape[1]
    if sparse.issparse(dms[0]):
        return X.tocsr()
    return X


def vstack(vecs, n, wavelength_dependence=None):
    mats = []
    for vec in vecs:
        vec_s = sparse.lil_matrix(vec)
        if wavelength_dependence is not None:
            mats.append(
                sparse.hstack(
                    [vec_s * wavelength_dependence[idx] for idx in np.arange(n)]
                )
            )
        else:
            mats.append(sparse.hstack([vec_s for idx in np.arange(n)]))
    return sparse.vstack(mats).T


def vstack_independent(mat, n):
    """Custom vertical stack script to stack lightkurve design matrices"""

    mat_s = sparse.lil_matrix(mat)
    npoints = mat.shape[0] * n
    ncomps = mat.shape[1] * n
    X = sparse.lil_matrix((npoints, ncomps))
    idx = 0
    jdx = 0
    for ndx in range(n):
        X[idx : idx + mat.shape[0], jdx : jdx + mat.shape[1]] = mat_s
        idx = idx + mat.shape[0]
        jdx = jdx + mat.shape[1]
    return X


def build_noise_matrix(visit):
    cdx = 0

    xs, ys, bkg = (
        visit.xshift[:, None] * np.ones((visit.nt, visit.nsp)),
        visit.yshift[:, None] * np.ones((visit.nt, visit.nsp)),
        visit.bkg[:, None] * np.ones((visit.nt, visit.nsp)),
    )
    Y = visit.Y[:, :, cdx]
    X = (
        (np.arange(visit.nt)[:, None, None] * np.ones(visit.T.shape) - visit.nt / 2)
        / (visit.nt)
    )[:, :, cdx]

    A = np.asarray([xs ** idx * ys ** jdx for idx in range(3) for jdx in range(2)][2:])
    A = np.vstack(
        [A, np.asarray([visit.T[:, :, cdx] ** idx for idx in np.arange(1, 2)])]
    )
    A = np.vstack(
        [A, np.asarray([X ** idx * Y ** jdx for idx in range(4) for jdx in range(4)])]
    )
    A1 = np.hstack(A.transpose([1, 0, 2])).T

    Anames = np.asarray(
        [f"$x_s^{idx}y_s^{jdx}$" for idx in range(3) for jdx in range(2)][2:]
    )
    Anames = np.hstack([Anames, np.asarray([f"$T^{idx}$" for idx in np.arange(1, 2)])])
    Anames = np.hstack(
        [
            Anames,
            np.asarray([f"$x^{idx}y^{jdx}$" for idx in range(4) for jdx in range(4)]),
        ]
    )

    fixed0 = vstack([ys.ravel()], visit.nwav)
    fixed1 = vstack(
        [ys.ravel()],
        visit.nwav,
        wavelength_dependence=np.linspace(-0.5, 0.5, visit.nwav),
    )

    noise = vstack_independent(A1, visit.nwav)
    As = sparse.hstack(
        [noise, fixed0, fixed1],
        format="csr",
    )
    Anames_all = np.asarray(
        [
            "$({})_{{{}}}$".format(name[1:-1], jdx)
            for jdx in range(visit.nwav)
            for name in Anames
        ]
    )
    Anames_all = np.hstack([Anames_all, "$ys_0$", "$ys_1$"])
    return Anames_all, As


def build_transit_matrix(visit, spline=False, nknots=30, knots=None, degree=3):

    no_ld_t = visit.no_limb_transit_subtime
    ld = visit.transit_subtime - no_ld_t
    eclipse = visit.eclipse_subtime
    if not spline:
        transit1 = vstack_independent(no_ld_t.ravel()[:, None], visit.nwav)
        eclipse1 = vstack_independent(eclipse.ravel()[:, None], visit.nwav)
    else:
        spline1 = lk.designmatrix.create_sparse_spline_matrix(
            visit.X[0, 0],
            n_knots=nknots,
            knots=knots,
            degree=degree,
        ).X
        spline = sparse.lil_matrix(
            (visit.nt * visit.nsp * visit.nwav, spline1.shape[1])
        )
        X = visit.X.transpose([2, 0, 1]).ravel()
        for idx, x in enumerate(visit.X[0, 0]):
            k = X == x
            spline[k] = spline1[idx]
        spline = spline.tocsr()
        t = sparse.csr_matrix(
            (visit.transit_subtime[:, :, None] * np.ones(visit.shape))
            .transpose([2, 0, 1])
            .ravel()
        ).T
        transit1 = spline.multiply(t)
        e = sparse.csr_matrix(
            (visit.eclipse_subtime[:, :, None] * np.ones(visit.shape))
            .transpose([2, 0, 1])
            .ravel()
        ).T
        eclipse1 = spline.multiply(e)

    fixed0 = vstack([ld.ravel()], visit.nwav)
    fixed1 = vstack(
        [ld.ravel()],
        visit.nwav,
        wavelength_dependence=np.linspace(-0.5, 0.5, visit.nwav),
    )
    As = sparse.hstack([transit1, eclipse1, fixed0, fixed1], format="csr")
    Anames = ["$\\delta f_{tr}$", "$\\delta f_{ec}$"]
    Anames_all = np.asarray(
        [
            "$({})_{{{}}}$".format(name[1:-1], jdx)
            for name in Anames
            for jdx in range(transit1.shape[1])
        ]
    )
    Anames_all = np.hstack([Anames_all, ["$u_0$", "$u_1$"]])

    return Anames, As


def fit_model(visit, spline: bool = False, nknots: int = 30, nsamps: int = 40):
    """
    Fits the eclipse/transit models for a given visit.

    Parameters
    ----------

    spline: bool
        Whether to use a spline model for the transit depth
        If True, will use splines. This will make the spectrum
        "smooth"
    nknots: int
        Number of knots for the spline
    nsamps: int
        Number of samples to draw for each spectrum
    """

    meta = visit.meta

    if spline:
        spline1 = lk.designmatrix.create_sparse_spline_matrix(
            visit.X[0, 0], n_knots=nknots
        ).X

    Anames_transit, As_transit = build_transit_matrix(
        visit, spline=spline, nknots=nknots
    )
    Anames_noise, As_noise = build_noise_matrix(visit)
    Anames, As = (np.hstack([Anames_transit, Anames_noise])), sparse.hstack(
        [As_transit, As_noise]
    )

    avg = visit.average_lc / visit.average_lc.mean()
    y = ((visit.data / visit.model) / avg[:, None, None]).transpose([2, 0, 1]).ravel()
    yerr = (
        ((visit.error / visit.model) / avg[:, None, None]).transpose([2, 0, 1]).ravel()
    )

    prior_sigma = np.ones(As.shape[1]) * 1000000

    oot = (visit.transit_subtime.mean(axis=1) + visit.eclipse_subtime.mean(axis=1)) == 0
    k = np.ones(visit.shape, bool).transpose([2, 0, 1]).ravel()
    for count in [0, 1]:
        # This makes the mask points have large errors
        c = (~k).astype(float) * 1e5 + 1
        sigma_w_inv = As.T.dot(As.multiply(1 / (yerr * c)[:, None] ** 2)).toarray()
        #            if jdx == 0:
        sigma_w_inv += np.diag(1 / prior_sigma ** 2)
        #            if jdx == 1:
        #                sigma_w_inv += prior_sigma_inv2d
        B = As.T.dot((y - y.mean()) / (yerr * c) ** 2)
        sigma_w = np.linalg.inv(sigma_w_inv)
        w = np.linalg.solve(sigma_w_inv, B)
        werr = sigma_w.diagonal() ** 0.5
        k &= np.abs(((y - y.mean()) - As.dot(w)) / yerr) < 5
    td = np.abs(visit.no_limb_transit_subtime.min())
    ed = np.abs(visit.eclipse_subtime.min())
    oot_flux = np.median(visit.average_lc[oot])

    # Package up result:
    if not spline:
        visit.transmission_spectrum = Spectrum(
            visit.wavelength.to(u.micron),
            1e6 * td * w[: visit.nwav] / oot_flux * ppm,
            1e6 * td * werr[: visit.nwav] / oot_flux * ppm,
            depth=td,
            name=visit.name + " Transmission Spectrum",
            visit=visit.visit_number,
            meta=meta,
        )
        visit.emission_spectrum = Spectrum(
            visit.wavelength.to(u.micron),
            1e6 * ed * w[visit.nwav : 2 * visit.nwav] / oot_flux * ppm,
            1e6 * ed * werr[visit.nwav : 2 * visit.nwav] / oot_flux * ppm,
            depth=ed,
            name=visit.name + " Emission Spectrum",
            visit=visit.visit_number,
            meta=meta,
        )

        tse = 1e6 * td * werr[: visit.nwav] / oot_flux
        ts = 1e6 * td * w[: visit.nwav] / oot_flux
        visit.transmission_spectrum_draws = Spectra(
            [
                Spectrum(
                    visit.wavelength.to(u.micron),
                    s1 * ppm,
                    tse * ppm,
                    depth=td,
                    name=visit.name + " Transmission Spectrum",
                    visit=visit.visit_number,
                    meta=meta,
                )
                for s1 in np.random.multivariate_normal(
                    ts,
                    np.diag(np.ones(visit.nwav)) * (tse) ** 2,
                    size=(nsamps,),
                )
            ],
            name=f"{visit.name} Transimission Spectrum Draws",
        )

        ese = 1e6 * ed * werr[visit.nwav : 2 * visit.nwav] / oot_flux
        es = 1e6 * ed * w[visit.nwav : 2 * visit.nwav] / oot_flux
        visit.emission_spectrum_draws = Spectra(
            [
                Spectrum(
                    visit.wavelength.to(u.micron),
                    s1 * ppm,
                    ese * ppm,
                    depth=ed,
                    name=visit.name + " Emission Spectrum",
                    visit=visit.visit_number,
                    meta=meta,
                )
                for s1 in np.random.multivariate_normal(
                    es,
                    np.diag(np.ones(visit.nwav)) * (ese) ** 2,
                    size=(nsamps,),
                )
            ],
            name=f"{visit.name} Emission Spectrum Draws",
        )

    else:
        samples = (
            1e6
            * td
            * spline1.dot(
                np.random.multivariate_normal(
                    w[:nknots], sigma_w[:nknots, :nknots], size=(nsamps,)
                ).T
            )
            / oot_flux
        )

        visit.transmission_spectrum_draws = Spectra(
            [
                Spectrum(
                    visit.wavelength.to(u.micron),
                    s1 * ppm,
                    s1 * np.nan * ppm,
                    depth=td,
                    name=visit.name + " Transmission Spectrum",
                    visit=visit.visit_number,
                    meta=meta,
                )
                for s1 in samples.T
            ],
            name=f"{visit.name} Transimission Spectrum Draws",
        )
        visit.transmission_spectrum = Spectrum(
            visit.wavelength.to(u.micron),
            1e6 * td * spline1.dot(w[:nknots]) / oot_flux * ppm,
            samples.std(axis=1) * ppm,
            depth=td,
            name=visit.name + " Transmission Spectrum",
            visit=visit.visit_number,
            meta=meta,
        )

        samples = (
            1e6
            * ed
            * spline1.dot(
                np.random.multivariate_normal(
                    w[nknots : nknots * 2],
                    sigma_w[nknots : nknots * 2, nknots : nknots * 2],
                    size=(nsamps,),
                ).T
            )
            / oot_flux
        )

        visit.emission_spectrum_draws = Spectra(
            [
                Spectrum(
                    visit.wavelength.to(u.micron),
                    s1 * ppm,
                    s1 * np.nan * ppm,
                    depth=ed,
                    name=visit.name + " Emission Spectrum",
                    visit=visit.visit_number,
                    meta=meta,
                )
                for s1 in samples.T
            ],
            name=f"{visit.name} Emission Spectrum Draws",
        )
        visit.emission_spectrum = Spectrum(
            visit.wavelength.to(u.micron),
            1e6 * ed * spline1.dot(w[nknots : 2 * nknots]) / oot_flux * ppm,
            1e6 * ed * spline1.dot(werr[nknots : 2 * nknots]) / oot_flux * ppm,
            depth=ed,
            name=visit.name + " Emission Spectrum",
            visit=visit.visit_number,
            meta=meta,
        )

    visit.full_model = (
        (
            (As.dot(w)).reshape((visit.nwav, visit.nt, visit.nsp)).transpose([1, 2, 0])
            + y.mean()
        )
        * avg[:, None, None]
        * visit.model
    )
