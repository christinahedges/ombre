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
        [A, np.asarray([X ** idx * Y ** jdx for idx in range(3) for jdx in range(3)])]
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


def build_transit_matrix(visit, spline=False):

    no_ld_t = visit.no_limb_transit_subtime
    ld = visit.transit_subtime - no_ld_t
    eclipse = visit.eclipse_subtime
    if not spline:
        transit1 = vstack_independent(
            no_ld_t[np.ones(no_ld_t.shape[:2], bool), :], visit.nwav
        )
        transit1 = sparse.hstack(
            [transit1[:, idx :: visit.nplanets] for idx in range(visit.nplanets)]
        )

        eclipse1 = vstack_independent(
            eclipse[np.ones(eclipse.shape[:2], bool), :], visit.nwav
        )
        eclipse1 = sparse.hstack(
            [eclipse1[:, idx :: visit.nplanets] for idx in range(visit.nplanets)]
        )

    else:
        spline1 = lk.designmatrix.create_sparse_spline_matrix(
            visit.wavelength.value,
            knots=np.arange(0.6, 1.8, 0.0075) * 1e4,
            degree=2,
        ).X
        spline = sparse.lil_matrix(
            (visit.nt * visit.nsp * visit.nwav, spline1.shape[1])
        )
        X = visit.X.transpose([2, 0, 1]).ravel()
        for idx, x in enumerate(visit.X[0, 0]):
            k = X == x
            spline[k] = spline1[idx]
        spline = spline.tocsr()
        transit1, eclipse1 = [], []
        for pdx in range(visit.nplanets):
            t = sparse.csr_matrix(
                (visit.transit_subtime[:, :, pdx][:, :, None] * np.ones(visit.shape))
                .transpose([2, 0, 1])
                .ravel()
            ).T
            transit1.append(spline.multiply(t))
            e = sparse.csr_matrix(
                (visit.eclipse_subtime[:, :, pdx][:, :, None] * np.ones(visit.shape))
                .transpose([2, 0, 1])
                .ravel()
            ).T
            eclipse1.append(spline.multiply(e))
        transit1 = sparse.hstack(transit1)
        eclipse1 = sparse.hstack(eclipse1)

    fixed0 = vstack(ld[np.ones(ld.shape[:2], bool), :].T, visit.nwav)
    fixed1 = vstack(
        ld[np.ones(ld.shape[:2], bool), :].T,
        visit.nwav,
        wavelength_dependence=np.linspace(-0.5, 0.5, visit.nwav),
    )
    As = sparse.hstack([transit1, eclipse1, fixed0, fixed1], format="csr")
    Anames = np.hstack(
        [
            [f"$\\delta f_{{tr, {letter}}}$" for letter in visit.letter],
            [f"$\\delta f_{{ec, {letter}}}$" for letter in visit.letter],
        ]
    )
    Anames_all = np.asarray(
        [
            "$({})_{{{}}}$".format(name[1:-1], jdx)
            for name in Anames
            for jdx in range(transit1.shape[1])
        ]
    )
    Anames_all = np.hstack([Anames_all, ["$u_0$", "$u_1$"]])

    return Anames, As


def fit_model(visit, spline: bool = False, nsamps: int = 40, suffix=""):
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

    if not hasattr(visit, "As_noise"):
        visit.Anames_noise, visit.As_noise = build_noise_matrix(visit)
    Anames_transit, As_transit = build_transit_matrix(visit, spline=spline)
    Anames, As = (np.hstack([Anames_transit, visit.Anames_noise])), sparse.hstack(
        [As_transit, visit.As_noise]
    )

    avg = visit.average_lc / visit.average_lc.mean()
    y = ((visit.data / visit.model) / avg[:, None, None]).transpose([2, 0, 1]).ravel()
    yerr = (
        ((visit.error / visit.model) / avg[:, None, None]).transpose([2, 0, 1]).ravel()
    )

    prior_sigma = np.ones(As.shape[1]) * 1000000

    oot = (
        visit.transit_subtime.sum(axis=-1).sum(axis=1)
        + visit.eclipse_subtime.sum(axis=-1).sum(axis=1)
    ) == 0
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
    tds = np.abs(visit.no_limb_transit_subtime.min(axis=(0, 1)))
    eds = np.abs(visit.eclipse_subtime.min(axis=(0, 1)))
    oot_flux = np.median(visit.average_lc[oot])
    # Package up result:
    if not hasattr(visit, "transmission_spectrum"):
        visit.transmission_spectrum, visit.emission_spectrum = {}, {}
    if not spline:
        for pdx in range(visit.nplanets):
            td, ed, letter, meta = (
                tds[pdx],
                eds[pdx],
                visit.letter[pdx],
                visit.meta(pdx),
            )

            visit.transmission_spectrum[
                f"{letter}_{suffix}" if suffix != "" else f"{letter}"
            ] = Spectrum(
                visit.wavelength.to(u.micron),
                1e6
                * td
                * w[pdx * visit.nwav : (pdx + 1) * visit.nwav]
                / oot_flux
                * ppm,
                1e6
                * td
                * werr[pdx * visit.nwav : (pdx + 1) * visit.nwav]
                / oot_flux
                * ppm,
                depth=td,
                name=visit.name + f"{letter} Transmission Spectrum",
                visit=visit.visit_number,
                meta=meta,
            )
            pdx += visit.nplanets
            visit.emission_spectrum[
                f"{letter}_{suffix}" if suffix != "" else f"{letter}"
            ] = Spectrum(
                visit.wavelength.to(u.micron),
                1e6
                * ed
                * w[pdx * visit.nwav : (pdx + 1) * visit.nwav]
                / oot_flux
                * ppm,
                1e6
                * ed
                * werr[pdx * visit.nwav : (pdx + 1) * visit.nwav]
                / oot_flux
                * ppm,
                depth=ed,
                name=visit.name + f"{letter} Emission Spectrum",
                visit=visit.visit_number,
                meta=meta,
            )

        # tse = 1e6 * td * werr[: visit.nwav] / oot_flux
        # ts = 1e6 * td * w[: visit.nwav] / oot_flux
        # visit.transmission_spectrum_draws = Spectra(
        #     [
        #         Spectrum(
        #             visit.wavelength.to(u.micron),
        #             s1 * ppm,
        #             tse * ppm,
        #             depth=td,
        #             name=visit.name + " Transmission Spectrum",
        #             visit=visit.visit_number,
        #             meta=meta,
        #         )
        #         for s1 in np.random.multivariate_normal(
        #             ts,
        #             np.diag(np.ones(visit.nwav)) * (tse) ** 2,
        #             size=(nsamps,),
        #         )
        #     ],
        #     name=f"{visit.name} Transimission Spectrum Draws",
        # )
        #
        # ese = 1e6 * ed * werr[visit.nwav : 2 * visit.nwav] / oot_flux
        # es = 1e6 * ed * w[visit.nwav : 2 * visit.nwav] / oot_flux
        # visit.emission_spectrum_draws = Spectra(
        #     [
        #         Spectrum(
        #             visit.wavelength.to(u.micron),
        #             s1 * ppm,
        #             ese * ppm,
        #             depth=ed,
        #             name=visit.name + " Emission Spectrum",
        #             visit=visit.visit_number,
        #             meta=meta,
        #         )
        #         for s1 in np.random.multivariate_normal(
        #             es,
        #             np.diag(np.ones(visit.nwav)) * (ese) ** 2,
        #             size=(nsamps,),
        #         )
        #     ],
        #     name=f"{visit.name} Emission Spectrum Draws",
        # )

    else:
        spline1 = lk.designmatrix.create_sparse_spline_matrix(
            visit.wavelength.value,
            knots=np.arange(1.1, 1.7, 0.01) * 1e4,
            degree=2,
        ).X
        nknots = spline1.shape[1]

        for pdx in range(visit.nplanets):
            td, ed, letter, meta = (
                tds[pdx],
                eds[pdx],
                visit.letter[pdx],
                visit.meta(pdx),
            )

            samples = (
                1e6
                * td
                * spline1.dot(
                    np.random.multivariate_normal(
                        w[pdx * nknots : (pdx + 1) * nknots],
                        sigma_w[
                            pdx * nknots : (pdx + 1) * nknots,
                            pdx * nknots : (pdx + 1) * nknots,
                        ],
                        size=(nsamps,),
                    ).T
                )
                / oot_flux
            )
            #
            # visit.transmission_spectrum_draws = Spectra(
            #     [
            #         Spectrum(
            #             visit.wavelength.to(u.micron),
            #             s1 * ppm,
            #             s1 * np.nan * ppm,
            #             depth=td,
            #             name=visit.name + " Transmission Spectrum",
            #             visit=visit.visit_number,
            #             meta=meta,
            #         )
            #         for s1 in samples.T
            #     ],
            #     name=f"{visit.name} Transimission Spectrum Draws",
            # )
            visit.transmission_spectrum[
                f"{letter}_{suffix}" if suffix != "" else f"{letter}"
            ] = Spectrum(
                visit.wavelength.to(u.micron),
                1e6
                * td
                * spline1.dot(w[pdx * nknots : (pdx + 1) * nknots])
                / oot_flux
                * ppm,
                samples.std(axis=1) * ppm,
                depth=td,
                name=visit.name + f"{letter} Transmission Spectrum",
                visit=visit.visit_number,
                meta=meta,
            )

            pdx += visit.nplanets
            samples = (
                1e6
                * ed
                * spline1.dot(
                    np.random.multivariate_normal(
                        w[pdx * nknots : (pdx + 1) * nknots],
                        sigma_w[
                            pdx * nknots : (pdx + 1) * nknots,
                            pdx * nknots : (pdx + 1) * nknots,
                        ],
                        size=(nsamps,),
                    ).T
                )
                / oot_flux
            )
            #
            # visit.emission_spectrum_draws = Spectra(
            #     [
            #         Spectrum(
            #             visit.wavelength.to(u.micron),
            #             s1 * ppm,
            #             s1 * np.nan * ppm,
            #             depth=ed,
            #             name=visit.name + "{letter} Emission Spectrum",
            #             visit=visit.visit_number,
            #             meta=meta,
            #         )
            #         for s1 in samples.T
            #     ],
            #     name=f"{visit.name} Emission Spectrum Draws",
            # )
            visit.emission_spectrum = Spectrum(
                visit.wavelength.to(u.micron),
                1e6
                * ed
                * spline1.dot(w[pdx * nknots : (pdx + 1) * nknots])
                / oot_flux
                * ppm,
                samples.std(axis=1) * ppm,
                depth=ed,
                name=visit.name + f"{letter} Emission Spectrum",
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
