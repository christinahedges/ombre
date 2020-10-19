import numpy as np

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
import matplotlib.pyplot as plt

import lightkurve as lk

def fit_transit(x, y, yerr, A, r_val, t0_val, period_val, inc_val, r_star, m_star, exptime,
                    sample=False, fit_period=True, fit_inc=False, fit_t0=True,
                     x_suppl=np.empty(0), y_suppl=np.empty(0), yerr_suppl=np.empty(0), exptime_suppl=None):
    breaks = np.where(np.diff(x) > 2)[0] + 1


    with pm.Model() as model:
        # The baseline flux
        norm = pm.Normal('norm', mu=y.mean(), sd=y.std(), shape=len(breaks) + 1)
        normalization = pm.Deterministic('normalization', tt.concatenate([norm[idx] + np.zeros_like(x) for idx, x in enumerate(np.array_split(x, breaks))]))

        if len(x_suppl) > 0:
            norm_suppl = pm.Normal('norm_suppl', mu=y_suppl.mean(), sd=y_suppl.std(), shape=1)


        # The time of a reference transit for each planet
        if fit_t0:
            t0 = pm.Uniform("t0", lower=t0_val-period_val/2, upper=t0_val+period_val/2, testval=t0_val)
        else:
            t0 = t0_val
        if fit_period:
            period = pm.Bound(pm.Normal, lower=period_val-0.01, upper=period_val+0.01)('period', mu=period_val, sd=0.0001)
        else:
            period = period_val

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        #u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
        u_suppl = pm.Uniform('u_suppl', lower=0, upper=1, testval=0.5, shape=1)
        u = pm.Uniform('u', lower=0, upper=1, testval=0.5, shape=1)

        r_star_pm = pm.Normal('r_star', mu=r_star, sd=0.1*r_star)
        r = pm.Normal(
            "r", mu=r_val, sd=r_val*0.3)
        ror = pm.Deterministic("ror", r / r_star_pm)
#        b = xo.ImpactParameter("b", ror=ror)
        if fit_inc:
            inc = pm.Normal('inc', mu=inc_val, sd=0.01)
        else:
            inc = inc_val

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, r_star=r_star_pm, m_star=m_star, incl=inc)

        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=x, texp=exptime
        )
        light_curve = pm.Deterministic('light_curve', (pm.math.sum(light_curves, axis=-1)) + 1)

        # Compute the model light curve using starry
        if len(x_suppl) > 0:
            r_suppl = pm.Normal(
                "r_suppl", mu=r_val, sd=r_val*0.3)

            light_curves_suppl = xo.LimbDarkLightCurve(u_suppl).get_light_curve(
                orbit=orbit, r=r_suppl, t=x_suppl, texp=exptime_suppl
            )
            light_curve_suppl = pm.Deterministic('light_curve_suppl', (pm.math.sum(light_curves_suppl, axis=-1)) + 1)


        sigma_w_inv = tt.dot(A.T, A/yerr[:, None]**2)
        B = tt.dot(A.T, (y - (light_curve * normalization))/yerr**2)
        w = tt.slinalg.solve(sigma_w_inv, B)
        noise_model = pm.Deterministic('noise_model', tt.dot(A, w))

        no_limb = pm.Deterministic('no_limb', pm.math.sum(xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(orbit=orbit, r=r, t=x, texp=exptime), axis=-1) + 1)

        full_model = pm.Deterministic('full_model', (light_curve * normalization) + noise_model)

        if len(x_suppl) == 0:
            pm.Normal("obs", mu=full_model, sd=yerr, observed=y)
        else:
            model_suppl = (light_curve_suppl * norm_suppl)
            pm.Normal("obs", mu=tt.concatenate([model_suppl, full_model]), sd=tt.concatenate([yerr_suppl, yerr]), observed=(tt.concatenate([y_suppl, y])))

        map_soln = xo.optimize(start=None, verbose=False)
        if sample:
            trace = xo.sample(
                tune=200, draws=1000, start=map_soln, chains=4, target_accept=0.95
            )
            return map_soln, pm.trace_to_dataframe(trace)
        return map_soln


def fit_white_light(obs, no_fit=False, fit_period=True, fit_t0=True, fit_inc=False, sample=False, supplement=None, t0_val=None, period_val=None, inc_val=None):
    if t0_val is not None:
        obs.sys.secondaries[0].t0 = t0_val
    if period_val is not None:
        obs.sys.secondaries[0].porb = period_val
    if inc_val is not None:
        obs.sys.secondaries[0].inc = inc_val

    t, lcs, lcs_err, orbits, xs, bkg = [], [], [], [], [], []
    for v in obs:
        t.append(v.time)
        lcs.append(v.average_lc/v.average_lc.mean())
        lcs_err.append(v.average_lc_errors/v.average_lc.mean())
        orbits.append(v.orbits)
        xs.append(v.xshift)
        bkg.append(v.bkg)

    x, y, yerr = np.hstack(t), np.hstack(lcs), np.hstack(lcs_err)

    breaks = np.where(np.diff(x) > 2)[0] + 1
    long_time = [(i - i.mean())/(i.max() - i.min()) for i in np.array_split(x, breaks)]
    orbit_trends = [np.hstack([(t[idx][o] - t[idx][o].min())/(t[idx][o].max() - t[idx][o].min()) - 0.5 for o in orbits[idx][:, np.any(orbits[idx], axis=0)].T]) for idx in range(len(t))]

    X = lambda x: np.vstack([np.atleast_2d(x[idx]).T * np.diag(np.ones(len(x)))[idx] for idx in range(len(x))])
    orbit = X(orbit_trends)
    background = X(bkg)
    xshift = X(xs)
    star = X(long_time)
    ones = X([x1**0 for x1 in xs])
    hook = X([-np.exp(-300 * (t1 - t1[0])) for t1 in t])
    hook2 = X([-np.exp(-100 * (t1 - t1[0])) for t1 in t])

    A = np.hstack([hook, hook2, xshift, background, background**2, ones, orbit, orbit**2, star, star**2])
    # Hacky garbage
    A = np.nan_to_num(A)
    A = np.asarray(A)

    if supplement is not None:
        if not isinstance(supplement, lk.LightCurve):
            raise TypeError('Pass a lightkurve object for the supplement keyword')
        x_suppl, y_suppl, yerr_suppl, exptime_suppl = (np.asarray(supplement.time, np.float64),
                                                        np.asarray(supplement.flux, np.float64),
                                                        np.asarray(supplement.flux_err, np.float64),
                                                        np.median(np.diff(supplement.time)))
    else:
        x_suppl, y_suppl, yerr_suppl, exptime_suppl = np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    if no_fit:
        model_lc = obs.sys.flux(x).eval()
        if model_lc.sum() == 0:
            warnings.warn('No transit detected')

        prior_sigma = np.hstack([hook[0]**0, hook2[0]**0, xshift[0]**0, background[0]**0, background[0]**0, ones[0]**0, orbit[0]**0, orbit[0]**0, star[0]**0, star[0]**0, 1])
        prior_mu = np.hstack([hook[0]*0, hook2[0]*0, xshift[0]*0, background[0]*0, background[0]*0, ones[0]**0, orbit[0]*0, orbit[0]*0, star[0]*0, star[0]*0, 0])

        A1 = np.hstack([A, model_lc[:, None] - 1])
        sigma_w_inv = A1.T.dot(A1/yerr[:, None]**2)
        sigma_w_inv += np.diag(1/prior_sigma**2)
        B = A1.T.dot(y/yerr**2)
        B += prior_mu/prior_sigma**2
        w = np.linalg.solve(sigma_w_inv, B)

        obs.model_lc = (model_lc[np.argsort(x)] - 1) * w[-1] + 1
        obs.model_lc_no_ld = (model_lc[np.argsort(x)] - 1) * w[-1] + 1
        obs.depth = -np.min(w[-1] * (model_lc - 1))
        obs.noise_model = A.dot(w[:-1])
        for idx, visit in enumerate(obs):
            visit.model_lc = (model_lc[np.in1d(x, t[idx])] - 1) * w[-1] + 1
            visit.model_lc_no_ld = (model_lc[np.in1d(x, t[idx])] - 1) * w[-1] + 1
            visit.noise_model = obs.noise_model[np.in1d(x, t[idx])]

    else:
        r = fit_transit(x, y, yerr, A,
                            r_val=obs.sys.secondaries[0].r.eval(),
                            t0_val=obs.sys.secondaries[0].t0.eval(), period_val=obs.sys.secondaries[0].porb.eval(),
                            inc_val=np.deg2rad(obs.sys.secondaries[0].inc.eval()),
                            r_star=obs.sys.primary.r.eval(), m_star=obs.sys.primary.m.eval(), exptime=np.median(obs.exptime) / (3600 * 24),
                            x_suppl=x_suppl, y_suppl=y_suppl, yerr_suppl=yerr_suppl, exptime_suppl=exptime_suppl,
                            fit_period=fit_period, fit_t0=fit_t0, sample=sample, fit_inc=fit_inc)

        if sample:
            map_soln, trace = r
            obs.wl_map_soln = map_soln
            obs.trace = trace
        else:
            map_soln = r
            obs.wl_map_soln = map_soln

        obs.model_lc = map_soln['light_curve']
        obs.model_lc_no_ld = map_soln['no_limb']
        obs.depth = (map_soln['r']/map_soln['r_star'])**2
        obs.noise_model = (map_soln['noise_model'] + 1) * map_soln['normalization']
        for idx, visit in enumerate(obs):
            visit.model_lc = obs.model_lc[np.in1d(x, t[idx])]
            visit.model_lc_no_ld = obs.model_lc[np.in1d(x, t[idx])]
            visit.noise_model = obs.noise_model[np.in1d(x, t[idx])]

        if 'period' in map_soln:
            obs.sys.secondaries[0].porb = map_soln['period']
        if 'inc' in map_soln:
            obs.sys.secondaries[0].inc = np.rad2deg(map_soln['inc'])
        if 't0' in map_soln:
            obs.sys.secondaries[0].t0 = map_soln['t0']
        obs.sys.secondaries[0].r = map_soln['r']
