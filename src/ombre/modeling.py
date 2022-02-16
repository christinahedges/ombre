"""Tools to help with modeling"""
import numpy as np
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import pymc3_ext as pmx
import matplotlib.pyplot as plt

import logging
import warnings

import matplotlib.pyplot as plt

#
#
# def fit_transit(
#     x,
#     y,
#     yerr,
#     r_val,
#     t0_val,
#     period_val,
#     inc_val,
#     r_star,
#     m_star,
#     exptime,
#     A,
#     offsets,
#     subtime=None,
#     fit_period=True,
#     fit_inc=False,
#     fit_t0=True,
#     x_suppl=np.empty(0),
#     y_suppl=np.empty(0),
#     yerr_suppl=np.empty(0),
#     u_suppl=None,
#     exptime_suppl=None,
#     calc_eclipse=True,
# ):
#     """Fits transit data using exoplanet
#
#     Parameters
#     ----------
#     x : np.ndarray
#         Time array
#     y : np.ndarray
#         Flux array
#     """
#     with warnings.catch_warnings():
#         # Stifle theano warnings
#         # Remove when Theano is updated
#         warnings.filterwarnings("ignore", category=DeprecationWarning)
#         warnings.filterwarnings("ignore", category=FutureWarning)
#
#         # Remove when arviz is updated
#         warnings.filterwarnings("ignore", category=UserWarning)
#
#         logger = logging.getLogger("theano.gof.compilelock")
#         logger.setLevel(logging.ERROR)
#         logger = logging.getLogger("theano.tensor.opt")
#         logger.setLevel(logging.ERROR)
#
#         if not np.isfinite(np.vstack([x_suppl, y_suppl, yerr_suppl])).all():
#             raise ValueError("Must pass finite values to `fit_transit`")
#
#         with pm.Model() as model:
#             if len(x_suppl) > 0:
#                 norm_suppl = pm.Normal("norm_suppl", mu=1, sigma=0.1, shape=1)
#
#             # The time of a reference transit for each planet
#             if fit_t0:
#                 t0 = pm.Uniform(
#                     "t0",
#                     lower=t0_val - period_val / 20,
#                     upper=t0_val + period_val / 20,
#                     testval=t0_val,
#                 )
#             else:
#                 t0 = t0_val
#             if fit_period:
#                 period = pm.Bound(
#                     pm.Normal, lower=period_val - 0.1, upper=period_val + 0.1
#                 )("period", mu=period_val, sigma=0.0001)
#             else:
#                 period = period_val
#
#             # The Kipping (2013) parameterization for quadratic limb darkening paramters
#             if u_suppl is None:
#                 u_suppl = xo.distributions.QuadLimbDark(
#                     "u_suppl", testval=np.array([0.4412, 0.2312])
#                 )
#             u = xo.distributions.QuadLimbDark("u", testval=np.array([0.4412, 0.2312]))
#
#             r_star_pm = pm.Normal(
#                 "r_star", mu=r_star, sigma=0.2 * r_star, testval=r_star
#             )
#             m_star_pm = pm.Normal(
#                 "m_star", mu=m_star, sigma=0.2 * m_star, testval=m_star
#             )
#             r = pm.Uniform("r", lower=r_val * 0.1, upper=r_val * 10, testval=r_val)
#             ror = pm.Deterministic("ror", r / r_star_pm)
#             #        b = xo.ImpactParameter("b", ror=ror)
#             if fit_inc:
#                 inc = pm.Bound(pm.Normal, lower=np.deg2rad(70), upper=np.deg2rad(90))(
#                     "inc",
#                     mu=inc_val,
#                     sigma=3,
#                     testval=inc_val - 1e-5,
#                     shape=inc_val.shape,
#                 )
#             else:
#                 inc = inc_val
#
#             # Set up a Keplerian orbit for the planets
#             orbit = xo.orbits.KeplerianOrbit(
#                 period=period, t0=t0, r_star=r_star_pm, m_star=m_star_pm, incl=inc
#             )
#
#             # Compute the model light curve using starry
#             transits = pm.Deterministic(
#                 "transits",
#                 xo.LimbDarkLightCurve(u).get_light_curve(
#                     orbit=orbit, r=r, t=x, texp=exptime
#                 ),
#             )
#             transit = pm.Deterministic("transit", (pm.math.sum(transits, axis=-1)))
#
#             if calc_eclipse:
#                 # Set up a Keplerian orbit for the planets
#                 eclipse_orbit = xo.orbits.KeplerianOrbit(
#                     period=period,
#                     t0=t0 + period / 2,
#                     r_star=r_star_pm,
#                     m_star=m_star_pm,
#                     incl=inc,
#                 )
#                 eclipse_r = pm.Uniform(
#                     "eclipse_r", lower=0, upper=0.1, testval=0.01, shape=len(r_val)
#                 )
#                 eclipse = pm.Deterministic(
#                     "eclipse",
#                     pm.math.sum(
#                         xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
#                             orbit=eclipse_orbit, r=eclipse_r, t=x, texp=exptime
#                         ),
#                         axis=-1,
#                     ),
#                 )
#
#             # Compute the model light curve using starry
#             if len(x_suppl) > 0:
#                 r_suppl = pm.Uniform(
#                     "r_suppl", lower=r_val * 0.1, upper=r_val * 10, testval=r_val
#                 )
#
#                 transits_suppl = xo.LimbDarkLightCurve(u_suppl).get_light_curve(
#                     orbit=orbit, r=r_suppl, t=x_suppl, texp=exptime_suppl
#                 )
#                 transit_suppl = pm.Deterministic(
#                     "transit_suppl", (pm.math.sum(transits_suppl, axis=-1))
#                 )
#                 if calc_eclipse:
#                     eclipse_r_suppl = pm.Uniform(
#                         "eclipse_r_suppl", lower=0, upper=r_val * 2, testval=r_val * 0.1
#                     )
#
#                     eclipses_suppl = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
#                         orbit=eclipse_orbit,
#                         r=eclipse_r_suppl,
#                         t=x_suppl,
#                         texp=exptime_suppl,
#                     )
#                     eclipse_suppl = pm.Deterministic(
#                         "eclipse_suppl", (pm.math.sum(eclipses_suppl, axis=-1))
#                     )
#
#             sigma_w_inv = tt.dot(A.T, A / yerr[:, None] ** 2)
#             if calc_eclipse:
#                 B = tt.dot(A.T, (y / (transit + eclipse + 1)) / yerr ** 2)
#             else:
#                 B = tt.dot(A.T, (y / (transit + 1)) / yerr ** 2)
#             w = tt.slinalg.solve(sigma_w_inv, B)
#             noise_model = pm.Deterministic("noise_model", tt.dot(A, w))
#
#             no_limb_transit = pm.Deterministic(
#                 "no_limb_transit",
#                 pm.math.sum(
#                     xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
#                         orbit=orbit, r=r, t=x, texp=exptime
#                     ),
#                     axis=-1,
#                 ),
#             )
#             if subtime is not None:
#                 transit_subtime = pm.Deterministic(
#                     "transit_subtime",
#                     pm.math.sum(
#                         xo.LimbDarkLightCurve(u).get_light_curve(
#                             orbit=orbit, r=r, t=subtime.ravel(), texp=exptime
#                         ),
#                         axis=-1,
#                     ),
#                 )
#                 if calc_eclipse:
#                     eclipse_subtime = pm.Deterministic(
#                         "eclipse_subtime",
#                         pm.math.sum(
#                             xo.LimbDarkLightCurve(u).get_light_curve(
#                                 orbit=eclipse_orbit,
#                                 r=eclipse_r,
#                                 t=subtime.ravel(),
#                                 texp=exptime,
#                             ),
#                             axis=-1,
#                         ),
#                     )
#                 no_limb_transit_subtime = pm.Deterministic(
#                     "no_limb_transit_subtime",
#                     pm.math.sum(
#                         xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
#                             orbit=orbit, r=r, t=subtime.ravel(), texp=exptime
#                         ),
#                         axis=-1,
#                     ),
#                 )
#
#             if calc_eclipse:
#                 full_model = pm.Deterministic(
#                     "full_model", (((transit + eclipse + 1) * noise_model))
#                 )
#             else:
#                 full_model = pm.Deterministic(
#                     "full_model", (((transit + 1) * noise_model))
#                 )
#
#             if len(x_suppl) == 0:
#                 pm.Normal("obs", mu=full_model, sigma=yerr, observed=y)
#             else:
#                 if calc_eclipse:
#                     model_suppl = (transit_suppl + eclipse_suppl + 1) * norm_suppl
#                     pm.Normal(
#                         "obs",
#                         mu=tt.concatenate([model_suppl, full_model]),
#                         sigma=tt.concatenate([yerr_suppl, yerr]),
#                         observed=(tt.concatenate([y_suppl, y])),
#                     )
#                 else:
#                     model_suppl = (transit_suppl + 1) * norm_suppl
#                     pm.Normal(
#                         "obs",
#                         mu=tt.concatenate([model_suppl, full_model]),
#                         sigma=tt.concatenate([yerr_suppl, yerr]),
#                         observed=(tt.concatenate([y_suppl, y])),
#                     )
#             map_soln = model.test_point
#             vars = [r]
#             for key in ["period", "t0", "inc"]:
#                 if locals()[f"fit_{key}"]:
#                     vars.append(locals()[f"{key}"])
#             if len(y_suppl) != 0:
#                 if not isinstance(u_suppl, list):
#                     vars.append(u_suppl)
#                 vars.append(r_suppl)
#
#             map_soln = pmx.optimize(
#                 start=map_soln,
#                 vars=vars,
#                 verbose=True,
#             )
#
#             vars = [r, u]
#             if calc_eclipse:
#                 vars.append(eclipse_r)
#             map_soln = pmx.optimize(
#                 start=map_soln,
#                 verbose=True,
#                 vars=vars,
#             )
#             map_soln = pmx.optimize(start=map_soln, verbose=True)
#             return model, map_soln


def fit_multi_transit(
    x,
    y,
    yerr,
    r_val,
    t0_val,
    period_val,
    inc_val,
    r_star,
    m_star,
    exptime,
    A,
    offsets,
    subtime=None,
    fit_period=True,
    fit_t0=True,
    x_suppl=np.empty(0),
    y_suppl=np.empty(0),
    yerr_suppl=np.empty(0),
    u_suppl=None,
    calc_eclipse=True,
    ttvs=False,
    letters=None,
    point=None,
    fit=True,
    sample=True,
    draws=200,
):
    """Fits transit data using exoplanet

    Parameters
    ----------
    x : np.ndarray
        Time array
    y : np.ndarray
        Flux array
    """
    if calc_eclipse and ttvs:
        raise ValueError("Choose TTVs or eclipses.")
    if letters is None:
        letters = np.arange(len(period_val))

    r_val = np.atleast_1d(r_val)
    t0_val = np.atleast_1d(t0_val)
    period_val = np.atleast_1d(period_val)
    inc_val = np.atleast_1d(inc_val)
    if x_suppl is not None:
        exptime_suppl = np.median(np.diff(x_suppl))
    if ttvs:
        fit_t0 = False
        fit_period = False
        t0s = []
        time = np.hstack([x, x_suppl]) if x_suppl is not None else x
        for p1, t01 in zip(period_val, t0_val):
            ts = np.arange(
                np.floor((time.min() - t01) / p1) - 1,
                np.ceil((time.max() - t01) / p1) + 1,
            )
            transit_times = [
                transit * p1
                for transit in ts
                if (np.abs((time - (t01 + transit * p1))) < 0.2).any()
            ]
            if len(transit_times) <= 1:
                raise ValueError("Not enough transits for a TTV orbit")
            t0s.append(transit_times)

    with warnings.catch_warnings():
        # Stifle theano warnings
        # Remove when Theano is updated
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Remove when arviz is updated
        warnings.filterwarnings("ignore", category=UserWarning)

        logger = logging.getLogger("theano.gof.compilelock")
        logger.setLevel(logging.ERROR)
        logger = logging.getLogger("theano.tensor.opt")
        logger.setLevel(logging.ERROR)

        if not np.isfinite(np.vstack([x_suppl, y_suppl, yerr_suppl])).all():
            raise ValueError("Must pass finite values to `fit_transit`")

        with pm.Model() as model:
            if y_suppl is not None:
                y_supplmean = pm.Normal("y_supplmean", mu=1, sd=0.01, testval=1)
            # The time of a reference transit for each planet
            if fit_t0:
                t0 = pm.Normal(
                    "t0",
                    mu=t0_val,
                    sigma=0.01,
                    testval=t0_val,
                    shape=len(t0_val),
                )
            else:
                t0 = t0_val
            if fit_period:
                period = pm.Normal(
                    "period",
                    mu=period_val,
                    sigma=0.05,
                    shape=len(period_val),
                    testval=period_val,
                )
            else:
                period = period_val

            # The Kipping (2013) parameterization for quadratic limb darkening paramters
            if u_suppl is None:
                u_suppl = xo.distributions.QuadLimbDark(
                    "u_suppl", testval=np.array([0.4412, 0.2312])
                )
            u = xo.distributions.QuadLimbDark("u", testval=np.array([0.4412, 0.2312]))

            r_star_pm = pm.Normal(
                "r_star", mu=r_star, sigma=0.2 * r_star, testval=r_star
            )
            m_star_pm = pm.Normal(
                "m_star", mu=m_star, sigma=0.2 * m_star, testval=m_star
            )
            r = pm.Uniform(
                "r",
                lower=r_val * 0.1,
                upper=r_val * 10,
                testval=r_val,
                shape=len(r_val),
            )
            b = xo.ImpactParameter("b", ror=r / r_star_pm, shape=len(period_val))
            # if fit_inc:
            #     inc = pm.Bound(pm.Normal, lower=np.deg2rad(75), upper=np.deg2rad(90))(
            #         "inc",
            #         mu=inc_val,
            #         sigma=5,
            #         testval=inc_val - 1e-5,
            #         shape=len(inc_val),
            #     )
            # else:
            #     inc = inc_val

            if ttvs:
                transit_times = []
                for i in range(len(t0s)):
                    transit_times.append(
                        pm.Normal(
                            f"tts_{letters[i]}",
                            mu=t0s[i],
                            sigma=0.01,
                            testval=t0s[i],
                            shape=len(t0s[i]),
                        )
                        + t0[i]
                    )
                orbit = xo.orbits.TTVOrbit(
                    transit_times=transit_times,
                    r_star=r_star_pm,
                    m_star=m_star_pm,
                    b=b,
                    period=period,
                )
            else:
                # Set up a Keplerian orbit for the planets
                orbit = xo.orbits.KeplerianOrbit(
                    period=period, t0=t0, r_star=r_star_pm, m_star=m_star_pm, b=b
                )

            # Compute the model light curve using starry
            transits = xo.LimbDarkLightCurve(u).get_light_curve(
                orbit=orbit, r=r, t=x, texp=exptime
            )

            if calc_eclipse:
                # Set up a Keplerian orbit for the planets
                eclipse_orbit = xo.orbits.KeplerianOrbit(
                    period=period,
                    t0=t0 + period / 2,
                    r_star=r_star_pm,
                    m_star=m_star_pm,
                    b=b,
                )
                eclipse_r = pm.Uniform("eclipse_r", lower=0, upper=0.1, testval=0.01)
                eclipses = xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
                    orbit=eclipse_orbit, r=eclipse_r, t=x, texp=exptime
                )

            # Compute the model light curve using starry
            if len(x_suppl) > 0:
                r_suppl = pm.Uniform(
                    "r_suppl",
                    lower=r_val * 0.1,
                    upper=r_val * 10,
                    testval=r_val,
                    shape=len(r_val),
                )

                transit_suppl = pm.math.sum(
                    xo.LimbDarkLightCurve(u_suppl).get_light_curve(
                        orbit=orbit,
                        r=r_suppl,
                        t=x_suppl,
                        texp=exptime_suppl,
                    ),
                    axis=-1,
                )

                if calc_eclipse:
                    eclipse_r_suppl = pm.Uniform(
                        "eclipse_r_suppl",
                        lower=0,
                        upper=r_val * 2,
                        testval=r_val * 0.1,
                        shape=len(r_val),
                    )
                    eclipse_suppl = pm.math.sum(
                        xo.LimbDarkLightCurve([0, 0]).get_light_curve(
                            orbit=eclipse_orbit,
                            r=eclipse_r_suppl,
                            t=x_suppl,
                            texp=exptime_suppl,
                        ),
                        axis=-1,
                    )

            sigma_w_inv = tt.dot(A.T, A / yerr[:, None] ** 2)
            if calc_eclipse:
                B = tt.dot(
                    A.T,
                    (
                        y
                        / (
                            pm.math.sum(transits, axis=-1)
                            + pm.math.sum(eclipses, axis=-1)
                            + 1
                        )
                    )
                    / yerr ** 2,
                )
            else:
                B = tt.dot(A.T, (y / (pm.math.sum(transits, axis=-1) + 1)) / yerr ** 2)
            w = tt.slinalg.solve(sigma_w_inv, B)
            noise_model = tt.dot(A, w)

            no_limb_transit = pm.math.sum(
                xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
                    orbit=orbit, r=r, t=x, texp=exptime
                ),
                axis=-1,
            )
            if subtime is not None:
                transit_subtime = xo.LimbDarkLightCurve(u).get_light_curve(
                    orbit=orbit, r=r, t=subtime.ravel(), texp=exptime
                )
                if calc_eclipse:
                    eclipse_subtime = xo.LimbDarkLightCurve(u).get_light_curve(
                        orbit=eclipse_orbit,
                        r=eclipse_r,
                        t=subtime.ravel(),
                        texp=exptime,
                    )
                no_limb_transit_subtime = xo.LimbDarkLightCurve(
                    np.asarray([0, 0])
                ).get_light_curve(orbit=orbit, r=r, t=subtime.ravel(), texp=exptime)

            if calc_eclipse:
                full_model = (
                    pm.math.sum(transits, axis=-1) + pm.math.sum(eclipses, axis=-1) + 1
                ) * noise_model
            else:
                full_model = (pm.math.sum(transits, axis=-1) + 1) * noise_model

            if len(x_suppl) == 0:
                pm.Normal("obs", mu=full_model, sigma=yerr, observed=y)
            else:
                if calc_eclipse:
                    model_suppl = transit_suppl + eclipse_suppl + y_supplmean
                    pm.Normal(
                        "obs",
                        mu=tt.concatenate([model_suppl, full_model]),
                        sigma=tt.concatenate([yerr_suppl, yerr]),
                        observed=(tt.concatenate([y_suppl, y])),
                    )
                else:
                    model_suppl = transit_suppl + y_supplmean
                    pm.Normal(
                        "obs",
                        mu=tt.concatenate([model_suppl, full_model]),
                        sigma=tt.concatenate([yerr_suppl, yerr]),
                        observed=(tt.concatenate([y_suppl, y])),
                    )
            map_soln = model.test_point if point is None else point
            vars = [r, u, b]
            if len(y_suppl) != 0:
                vars.append(r_suppl)
                vars.append(u_suppl)
                vars.append(y_supplmean)
            if calc_eclipse:
                vars.append(eclipse_r)

            if fit:
                map_soln = pmx.optimize(
                    start=map_soln,
                    vars=vars,
                    verbose=True,
                )
            if sample:
                map_soln = pmx.optimize(start=map_soln, verbose=True)
                trace = pmx.sample(
                    tune=np.max([draws, 150]),
                    draws=draws,
                    start=map_soln,
                    cores=2,
                    chains=2,
                    target_accept=0.9,
                    return_inferencedata=True,
                )
            else:
                trace = None
        # I'm fixing these because I don't want them to be sampled for now...
        attrs = np.asarray(
            [
                "no_limb_transit_subtime",
                "transit_subtime",
                "transits",
                "noise_model",
            ]
        )
        if calc_eclipse:
            attrs = np.hstack([attrs, "eclipse_subtime", "eclipses"])
        result = {}
        for attr in attrs:
            result[attr] = pmx.eval_in_model(
                locals()[attr], model=model, point=map_soln
            )

        return model, map_soln, trace, result