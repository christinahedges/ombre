"""Tools to help with modeling"""
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
import pymc3_ext as pmx
import matplotlib.pyplot as plt


def fit_transit(
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
    subtime=None,
    fit_period=True,
    fit_inc=False,
    fit_t0=True,
    x_suppl=np.empty(0),
    y_suppl=np.empty(0),
    yerr_suppl=np.empty(0),
    exptime_suppl=None,
    npoly=2,
):
    """Fits transit data using exoplanet

    Parameters
    ----------
    x : np.ndarray
        Time array
    y : np.ndarray
        Flux array
    """
    breaks = np.where(np.diff(x) > 2)[0] + 1
    # if len(breaks) == 0:
    #     breaks = np.atleast_1d(len(x))
    # poly = np.vstack(
    #     [
    #         (x * np.in1d(x, x1) - (x * np.in1d(x, x1)).mean())
    #         / ((x * np.in1d(x, x1)).max() - (x * np.in1d(x, x1)).min())
    #         for idx, x1 in enumerate(np.array_split(x, breaks))
    #         if len(x1) != 0
    #     ]
    # ).T
    #
    # x_broken = np.vstack(
    #     [
    #         (x * np.in1d(x, x1) - x[np.in1d(x, x1)][0])
    #         for idx, x1 in enumerate(np.array_split(x, breaks))
    #         if len(x1) != 0
    #     ]
    # ).T
    #
    # if A is None:
    #     A = np.hstack([poly ** idx for idx in np.arange(0, npoly + 1)])
    # else:
    #     A = np.hstack([A, np.hstack([poly ** idx for idx in np.arange(0, npoly + 1)])])

    with pm.Model() as model:
        # # The baseline flux
        norm = pm.Normal("norm", mu=y.mean(), sigma=y.std(), shape=len(breaks) + 1)
        normalization = pm.Deterministic(
            "normalization",
            tt.concatenate(
                [
                    norm[idx] + np.zeros_like(x)
                    for idx, x in enumerate(np.array_split(x, breaks))
                ]
            ),
        )

        # hook_coeffs = pm.Uniform(
        #     "hook_coeffs",
        #     lower=-5,
        #     upper=1,
        #     shape=(len(breaks),),
        #     testval=np.zeros(len(breaks)),
        # )
        # hook_exps = pm.Uniform(
        #     "hook_exps",
        #     lower=-500,
        #     upper=-100,
        #     shape=(len(breaks),),
        #     testval=np.ones(len(breaks)) * -300,
        # )
        # hook_coeffs = np.zeros(len(breaks))
        # hook_exps = np.ones(len(breaks)) * -300
        # hook = pm.Deterministic(
        #     "hook", tt.sum(-(10 ** hook_coeffs) * tt.exp(hook_exps * x_broken), axis=1)
        # )

        if len(x_suppl) > 0:
            norm_suppl = pm.Normal(
                "norm_suppl", mu=y_suppl.mean(), sigma=y_suppl.std(), shape=1
            )

        # The time of a reference transit for each planet
        if fit_t0:
            t0 = pm.Uniform(
                "t0",
                lower=t0_val - period_val / 2,
                upper=t0_val + period_val / 2,
                testval=t0_val,
            )
        else:
            t0 = t0_val
        if fit_period:
            period = pm.Bound(
                pm.Normal, lower=period_val - 0.1, upper=period_val + 0.1
            )("period", mu=period_val, sigma=0.0001)
        else:
            period = period_val

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        u_suppl = xo.distributions.QuadLimbDark("u_suppl", testval=np.array([0.3, 0.2]))
        u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
        # u_suppl = pm.Uniform("u_suppl", lower=0, upper=1, testval=0.5, shape=1)
        # u = pm.Uniform("u", lower=0, upper=1, testval=0.5, shape=1)

        r_star_pm = pm.Normal("r_star", mu=r_star, sigma=0.1 * r_star, testval=r_star)
        r = pm.Uniform("r", lower=r_val * 0.1, upper=r_val * 10, testval=r_val)
        ror = pm.Deterministic("ror", r / r_star_pm)
        #        b = xo.ImpactParameter("b", ror=ror)
        if fit_inc:
            inc = pm.Normal("inc", mu=inc_val, sigma=0.1, testval=inc_val)
        else:
            inc = inc_val

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=t0, r_star=r_star_pm, m_star=m_star, incl=inc
        )

        # Compute the model light curve using starry
        transits = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=x, texp=exptime
        )
        transit = pm.Deterministic("transit", (pm.math.sum(transits, axis=-1)))

        # Set up a Keplerian orbit for the planets
        eclipse_orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=t0 + period / 2, r_star=r_star_pm, m_star=m_star, incl=inc
        )
        eclipse_r = pm.Uniform("eclipse_r", lower=0, upper=0.1, testval=0.01)
        eclipse = pm.Deterministic(
            "eclipse",
            pm.math.sum(
                xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
                    orbit=eclipse_orbit, r=eclipse_r, t=x, texp=exptime
                ),
                axis=-1,
            ),
        )

        # Compute the model light curve using starry
        if len(x_suppl) > 0:
            r_suppl = pm.Uniform(
                "r_suppl", lower=r_val * 0.1, upper=r_val * 10, testval=r_val
            )

            eclipse_r_suppl = pm.Uniform(
                "r_suppl", lower=0, upper=r_val * 2, testval=r_val * 0.1
            )

            transits_suppl = xo.LimbDarkLightCurve(u_suppl).get_light_curve(
                orbit=orbit, r=r_suppl, t=x_suppl, texp=exptime_suppl
            )
            transit_suppl = pm.Deterministic(
                "transit_suppl", (pm.math.sum(transits_suppl, axis=-1))
            )

            eclipses_suppl = xo.LimbDarkLightCurve([0, 0]).get_light_curve(
                orbit=eclipse_orbit, r=r_suppl, t=x_suppl, texp=exptime_suppl
            )
            eclipse_suppl = pm.Deterministic(
                "transit_suppl", (pm.math.sum(eclipses_suppl, axis=-1))
            )

        sigma_w_inv = tt.dot(A.T, A / yerr[:, None] ** 2)
        B = tt.dot(A.T, (y - ((transit + eclipse) * normalization)) / yerr ** 2)
        w = tt.slinalg.solve(sigma_w_inv, B)
        noise_model = pm.Deterministic("noise_model", tt.dot(A, w))

        no_limb_transit = pm.Deterministic(
            "no_limb_transit",
            pm.math.sum(
                xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
                    orbit=orbit, r=r, t=x, texp=exptime
                ),
                axis=-1,
            ),
        )
        if subtime is not None:
            transit_subtime = pm.Deterministic(
                "transit_subtime",
                pm.math.sum(
                    xo.LimbDarkLightCurve(u).get_light_curve(
                        orbit=orbit, r=r, t=subtime.ravel(), texp=exptime
                    ),
                    axis=-1,
                ),
            )
            eclipse_subtime = pm.Deterministic(
                "eclipse_subtime",
                pm.math.sum(
                    xo.LimbDarkLightCurve(u).get_light_curve(
                        orbit=eclipse_orbit,
                        r=eclipse_r,
                        t=subtime.ravel(),
                        texp=exptime,
                    ),
                    axis=-1,
                ),
            )
            no_limb_transit_subtime = pm.Deterministic(
                "no_limb_transit_subtime",
                pm.math.sum(
                    xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(
                        orbit=orbit, r=r, t=subtime.ravel(), texp=exptime
                    ),
                    axis=-1,
                ),
            )

        full_model = pm.Deterministic(
            "full_model", ((transit + eclipse + noise_model) * normalization)
        )

        if len(x_suppl) == 0:
            pm.Normal("obs", mu=full_model, sigma=yerr, observed=y)
        else:
            model_suppl = transit_suppl * norm_suppl
            pm.Normal(
                "obs",
                mu=tt.concatenate([model_suppl, full_model]),
                sigma=tt.concatenate([yerr_suppl, yerr]),
                observed=(tt.concatenate([y_suppl, y])),
            )

        map_soln = pmx.optimize(start=model.test_point, verbose=True, vars=[norm])
        #        map_soln = pmx.optimize(start=map_soln, verbose=True, vars=[hook])
        map_soln = pmx.optimize(start=map_soln, verbose=True)
        return model, map_soln
