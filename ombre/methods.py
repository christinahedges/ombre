"""Common ombre methods"""
import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
from matplotlib import animation
import warnings

from astropy.io import fits
from astropy.modeling.blackbody import blackbody_lambda
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.convolution import convolve, Gaussian2DKernel

from scipy.optimize import minimize
from scipy import sparse

import lightkurve as lk

from fbpca import pca

from tqdm.notebook import tqdm

from scipy.stats import pearsonr

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
from numba import jit

from . import PACKAGEDIR
CALIPATH = '{}{}'.format(PACKAGEDIR, '/data/calibration/')


def animate(data, scale='linear', output='out.mp4', **kwargs):
    '''Create an animation of all the frames in `data`.

    Parameters
    ----------
    data : np.ndarray
        3D np.ndarray
    output : str
        File to output mp4 to
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    idx = 0
    if scale is 'log':
        dat = np.log10(np.copy(data))
    else:
        dat = data
    cmap = kwargs.pop('cmap', 'Greys_r')
    cmap = plt.get_cmap(cmap)
    cmap.set_bad('black')
    if 'vmax' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(dat, 70)
        kwargs['vmax'] = np.nanpercentile(dat, 99.9)
    im = ax.imshow(dat[idx], origin='bottom', cmap=cmap,
                   **kwargs)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.axis('off')

    def animate(idx):
        im.set_data(dat[idx])
        return im,

    anim = animation.FuncAnimation(fig, animate, frames=len(
        dat), interval=30, blit=True)
    anim.save(output, dpi=150)


def simple_mask(sci, filter='G141'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = np.atleast_3d(sci.mean(axis=0) > np.nanpercentile(sci, 50)).transpose([2, 0, 1])
        #return mask
        data = sci/mask
        data[~np.isfinite(data)] = np.nan

        spectral = np.nanmean(data, axis=(0, 1))
        spatial = np.nanmean(data, axis=(0, 2))

        spatial_cut = 0.2
        if filter == 'G102':
            spectral_cut = 0.1
        if filter == 'G141':
            spectral_cut = 0.3

        spectral = spectral > np.nanmax(spectral) * spectral_cut
        if filter == 'G102':
            edge = np.where((np.gradient(spectral/np.nanmax(spectral)) < -0.07))[0][0] - 10
            m = np.ones(len(spectral), bool)
            m[edge:] = False
            spectral &= m

        spatial = spatial > np.nanmax(spatial) * spatial_cut

        mask = (np.atleast_3d(np.atleast_2d(spectral) * np.atleast_2d(spatial).T)).transpose([2, 0, 1])

        # Cut out the zero phase dispersion!
        # Choose the widest block as the true dispersion...
        secs = mask[0].any(axis=0).astype(int)
        len_secs = np.diff(np.where(np.gradient(secs) != 0)[0][::2] + 1)[::2]
        if len(len_secs) > 1:
            #a = (np.where(np.gradient(secs) != 0)[0][::2] + 1)[np.argmax(len_secs)*2:np.argmax(len_secs)*2+2]
            #print(a)
            #b = ((np.in1d(np.arange(len(mask)), np.arange(a[0] - 20, a[1] + 20)))[:, None] * np.ones(len(mask))).T
            #print(b)
            mask[:, :, :np.where(np.gradient(secs) > 0)[0][3]] &= False
            spectral = mask[0].any(axis=0)

    return mask, np.atleast_3d(np.atleast_2d(spectral)).transpose([2, 0, 1]), np.atleast_3d(np.atleast_2d(spatial.T)).transpose([2, 1, 0])

def average_vsr(obs):
    avg = np.atleast_3d(np.average(obs.data, weights=1/obs.error, axis=1)).transpose([0, 2, 1])
    vsr_mean = np.asarray([np.average(obs.data[idx]/avg[idx], weights=avg[idx]/obs.error[idx], axis=1) for idx in range(obs.nt)])
    vsr_mean = np.atleast_3d(vsr_mean) * np.ones(obs.shape)
    vsr_mean = (vsr_mean.T/np.mean(vsr_mean, axis=(1, 2))).T
    return vsr_mean


def average_spectrum(obs):
    if hasattr(obs, 'vsr_mean'):
        avg = np.atleast_3d(obs.vsr_mean)
    else:
        avg = np.atleast_3d(np.atleast_3d(np.average(obs.data, weights=1/obs.error, axis=2)).transpose([0, 2, 1]))
    spec_mean = np.average(obs.data/avg, weights=avg/obs.error, axis=(0, 1))
    spec_mean /= np.mean(spec_mean)
    return (np.atleast_3d(spec_mean) * np.ones(obs.shape).transpose([0, 2, 1])).transpose([0, 2, 1])

def get_flatfield(obs, visit, model):
    ''' Get flat field'''
    g141_flatfile = CALIPATH+'WFC3.IR.G141.flat.2.fits'
    ff_hdu = fits.open(g141_flatfile)

    nt, nrow, ncol = visit.shape


    norm = np.nansum(visit.data, axis=(1, 2))
    norm /= np.median(norm)
    norm = np.atleast_3d(norm).transpose([1, 0, 2])

    f = np.asarray([ff_hdu[i].data for i in range(4)])
    f = f[:, 1014//2 - obs.ns//2 : 1014//2 + obs.ns//2, 1014//2 - obs.ns//2 : 1014//2 + obs.ns//2]

    f = np.vstack([f, np.ones((1, f.shape[1], f.shape[2]))])
    X = f[:, visit.spatial.reshape(-1)][:, :, visit.spectral.reshape(-1)][:, np.ones((nrow, ncol), bool)].T
    avg = np.nanmean((visit.data/norm/model), axis=0)[np.ones((nrow, ncol), bool)]
    avg_err = ((1/nt)*np.nansum(((visit.error/norm/model))**2, axis=0)**0.5)[np.ones((nrow, ncol), bool)]

    sigma_w_inv = np.dot(X.T, X/avg_err[:, None]**2)
    B = np.dot(X.T, (avg/avg_err**2)[:, None])
    w = np.linalg.solve(sigma_w_inv, B)[:, 0]

    X = f[:, np.ones((obs.ns, obs.ns), bool)].T
    flat = np.dot(X, w).reshape((obs.ns, obs.ns))
    flat /= np.median(flat)
    flat[(obs.dq[0] & 256) != 0] = 1
    flat = np.atleast_3d(flat).transpose([2, 0, 1])
    return flat



def literature_calibrate(obs):
    for visit in obs:
        t = visit.time[0] - obs.dimage_time
        t[t < 0] = np.nan
        l = np.nanargmin(t)

        dcrpix1, dcrpix2 = obs.dimage_crpix1[l], obs.dimage_crpix2[l]
        src_x, src_y = obs.dimage_x[l], obs.dimage_y[l]
        dx, dy = visit.crpix1[0] - dcrpix1, visit.crpix2[0] - dcrpix2

        Y, X = np.mgrid[:visit.mask.shape[1], :visit.mask.shape[2]]
        xp = (1024/2 - visit.mask.shape[1]/2) + visit.crpix1[0] + visit.postarg1[0]
        yp = (1024/2 - visit.mask.shape[1]/2) + visit.crpix2[0]
        m = visit.mask[0].astype(float)
        m[m == 0] = np.nan
        yc = np.nanmean((Y * m))
        xc = np.nanmean((X * m), axis=(0))
        d = (xc - dx - src_x)
        xp = src_x + (1024/2 - visit.mask.shape[1]/2) + dcrpix1
        yp = src_y + (1024/2 - visit.mask.shape[1]/2) + dcrpix2
        if visit.filters[0] == 'G141':
            Bc = [8.95431E3,  9.35925E-2, 0]
            Bm = [4.51423E1, 3.17239E-4, 2.17055E-3, -7.42504E-7, 3.48639E-7, 3.09213E-7]
        elif visit.filters[0] == 'G102':
            Bc = [6.38738E3, 4.55507E-2, 0]
            Bm = [2.35716E1, 3.60396E-4, 1.58739E-3, -4.25234E-7, -6.53726E-8, 0]
        else:
            raise ValueError('Please pass G141 or G102 images.')
        c = Bc[0] + Bc[1] * xp + Bc[2] * yp
        m = Bm[0] + Bm[1] * xp + Bm[3] * xp**2 + Bm[4] * yp * xp + Bm[5] * yp**2 + Bm[2] * yp
        visit.wavelength = (m * d + c)[visit.mask[0].any(axis=0)]


def calibrate(obs, teff=5500, plot=False):
    filter = obs.filters[0]
    g141_sensfile = CALIPATH+f'WFC3.IR.{filter}.1st.sens.2.fits'

    hdu = fits.open(g141_sensfile)
    data = np.copy(obs.trace)
    data /= np.median(data)


    cent = np.average(np.arange(data.shape[0]), weights=data)

    for count in [0, 1, 2]:
        sens_raw = hdu[1].data['SENSITIVITY']
        wav = hdu[1].data['WAVELENGTH'] * u.angstrom
        bb = blackbody_lambda(wav, teff)
        bb /= np.trapz(bb, wav)
        sens = sens_raw * bb.value
        sens /= np.median(sens)

        if filter == 'G141':
            dw = 17000 - 10500
        else:
            dw = 11500 - 7700
        w_mean = np.mean(wav.value)

        g1 = np.gradient(sens)
        g1[:100] = 0
        g1[-100:] = 0
        if filter == 'G141':
            sens[np.argmax(g1)+300:np.argmin(g1)-300] = np.nan

        def func(params, return_model=False):
            model = np.interp(np.arange(0, data.shape[0]), (wav.value - w_mean)/dw *  params[0] + params[1], sens)
            if return_model:
                return model
            return np.nansum((obs.spec_mean[0, 0] - model)**2)/(np.isfinite(model).sum())
        na = 200
        nb = 201
        if filter == 'G141':
            a = np.linspace((data.shape[0])*0.85, (data.shape[0])*1.1, na)
            b = np.linspace(cent - 3, cent + 3, nb)
        if filter == 'G102':
            a = np.linspace((data.shape[0])*0.85, (data.shape[0])*1.1, na)
            b = np.linspace(cent - 7, cent + 7, nb)
        chi = np.zeros((na, nb))
        for idx, a1 in enumerate(a):
            for jdx, b1 in enumerate(b):
                model = func([a1, b1], return_model=True)
                if filter == 'G102':
                    model[np.where(data > 0.6)[0][0]: np.where(np.gradient(data) < -0.1)[0][0] - 5] = np.nan
                chi[idx, jdx] = np.nansum((data - model)**2)/(np.isfinite(model).sum())
        l = np.unravel_index(np.argmin(chi), chi.shape)
        params = [a[l[0]], b[l[1]]]
        wavelength = ((np.arange(data.shape[0]) - params[1]) / params[0]) * dw + w_mean
        sens = sens_raw #* bb.value[10:-10]
        sens /= np.median(sens)

        sensitivity = np.interp(wavelength, wav, sens)
        def func(teff):
            bb = blackbody_lambda(wavelength, teff)
            bb /= np.trapz(bb, wavelength)
            sens1 = sensitivity * bb.value
            sens1 /= np.median(sens1)
            return np.nansum((data - sens1)**2/data)
        r = minimize(func, [4000], method='TNC', bounds=[(1000, 20000)])
        teff = r.x[0]

    bb = blackbody_lambda(wavelength, teff)
    bb /= np.trapz(bb, wavelength)
    sensitivity_t = sensitivity * bb.value
    sensitivity_t /= np.median(sensitivity_t)

    if plot:
        plt.figure()
        plt.pcolormesh(a, b, chi.T, cmap='coolwarm')
        plt.colorbar()
        plt.scatter(*params, c='lime')
        plt.scatter(data.shape[0], cent, c='orange', marker='x')
    return wavelength, sensitivity, sensitivity_t



def build_vsr(obs, errors=False):
    """ Build a full model of the vsr

    Parameters
    ----------
    obs : shadow.Observation
        Input observation

    Returns
    -------
    vsr_grad_model : np.ndarray
        Array of the mean variable scan rate shifts
    """

    frames = obs.data / (obs.model)
    frames_err = obs.error  / (obs.model)

    frames_err = (frames_err.T/np.average(frames, weights=1/obs.error, axis=(1, 2))).T
    frames = (frames.T/np.average(frames, weights=1/obs.error, axis=(1, 2))).T
    frames[(frames_err/frames) > 0.1] = 1


    model = np.zeros(obs.shape)
    for tdx in tqdm(range(obs.nt)):
        A, _, _ = pca(frames[tdx], k=1, n_iter=100)
        A = np.hstack([A, np.atleast_2d(np.ones(obs.nsp)).T])
        A = np.vstack([(np.atleast_2d(a).T * np.ones(frames[0].shape)).ravel() for a in A.T]).T
        A = np.hstack([A,
                       A * np.atleast_2d(obs.X[0].ravel()).T,
                       A * np.atleast_2d(obs.Y[0].ravel()).T,
                       A * np.atleast_2d((obs.X[0] * obs.Y[0]).ravel()).T,])
        cadence_mask = (obs.spec_mean[tdx] > 0.6).ravel()
        sigma_w_inv = A[cadence_mask].T.dot(A[cadence_mask]/(frames_err[tdx].ravel()**2)[cadence_mask, None])
        B = A[cadence_mask].T.dot((frames[tdx]/frames_err[tdx]**2).ravel()[cadence_mask])
        w = np.linalg.solve(sigma_w_inv, B)
        model[tdx] = A.dot(w).reshape(frames[tdx].shape)
    return model



def build_spectrum(obs, errors=False, npoly=3):

    def _make_A(obs):
        diags = np.diag(np.ones(obs.shape[1]))
        blank = sparse.csr_matrix(np.ones(obs.shape[:2]))
        rows = sparse.vstack([sparse.csr_matrix((d * np.ones(obs.shape[:2])).ravel()) for d in diags]).T

        xshift = obs.xshift
        xshift -= np.mean(xshift)
        xshift /= (np.max(xshift) - np.min(xshift))
        xshift = np.atleast_3d(xshift).transpose([1, 0, 2]) * np.ones(obs.shape)
        t2 = sparse.csr_matrix(xshift[:, :, 0].ravel()).T

        y, t = sparse.csr_matrix(obs.Y[:, :, 0].ravel() - obs.Y.min()).T, sparse.csr_matrix(obs.T[:, :, 0].ravel() - obs.T.min()).T

        ones = sparse.csr_matrix(np.ones(y.shape[0])).T
        wlc = sparse.csr_matrix((np.atleast_2d(obs.average_lc - 1).T * np.ones(obs.shape[:2])).ravel()).T
        delta_depth = sparse.csr_matrix((np.atleast_2d((obs.model_lc != 1).astype(float)).T * np.ones(obs.shape[:2])).ravel()).T


        def poly(x, npoly=2):
            mul = lambda x: x.multiply(x)
            r = sparse.csr_matrix(np.ones(x.shape[0])).T
            r = sparse.hstack([r, x], format='csr')
            for i in range(npoly - 2):
                r = sparse.hstack([r, mul(r[:, -1])], format='csr')
            return r

        At = poly(t, npoly - 1)
        At2 = poly(t2, npoly - 1)
        Ay = poly(y, npoly)

        A = sparse.hstack([(At.T.multiply(A)).T for A in Ay.T])
        A1 = sparse.hstack([(At2[:, 1:].T.multiply(A)).T for A in Ay.T])

        A = sparse.hstack([wlc, delta_depth, A, A1])

        prior_mu = np.zeros(A.shape[1])
        prior_sigma = np.ones(A.shape[1]) * 0.1
        prior_mu[0] = 1

        A = sparse.hstack([A, rows], format='csr')

        prior_mu = np.hstack([prior_mu, np.ones(rows.shape[1])])
        prior_sigma = np.hstack([prior_sigma, np.ones(rows.shape[1]) * 0.1])

        return A, prior_mu, prior_sigma

    frames = obs.data / (obs.model)
    frames_err = obs.error  / (obs.model)
    frames_err /= np.median(frames)
    frames /= np.median(frames)

    A1, prior_mu1, prior_sigma1 = _make_A(obs)

    y_model = np.zeros(obs.shape)
    if errors:
        y_model_errs = np.zeros(obs.shape)

    ws = []
    masks = np.atleast_3d(obs.vsr_mean.mean(axis=0) > 0.93).transpose([2, 0, 1]) * np.ones(obs.shape)
    for idx in tqdm(range(obs.shape[2]), desc='Building Spectrum Model'):
        vsr_trend  = sparse.csr_matrix((obs._vsr_grad_model[:, :, idx].T * obs.xshift).T.ravel()).T
        A = sparse.hstack([A1, vsr_trend], format='csr')
        prior_mu = np.hstack([prior_mu1, 0])
        prior_sigma = np.hstack([prior_sigma1, 0.1])

        frame = frames[:, :, idx]
        frame_err = frames_err[:, :, idx]

        y = (frame).ravel()
        ye = (frame_err).ravel()
        mask = masks.astype(bool)[:, :, idx].ravel()#(obs.vsr_mean[:, :, idx] > 0.93).ravel()

        # Linear algebra to find the best fitting shifts
        sigma_w_inv = A[mask].T.dot(A[mask]/ye[mask, None]**2)
        sigma_w_inv += np.diag(1. / prior_sigma**2)
        B = A[mask].T.dot(y[mask]/ye[mask]**2)
        B += (prior_mu / prior_sigma**2)
        w = np.linalg.solve(sigma_w_inv, B)
        ws.append(w)
        m = A.dot(w).reshape(frame.shape)
        y_model[:, :, idx] = A[:, 2:].dot(w[2:]).reshape(frame.shape)
        y_model[:, :, idx] -= y_model[:, :, idx].mean()
        y_model[:, :, idx] += m.mean()

        #import pdb;pdb.set_trace()

        if errors:
#            y_model_errs[:, :, idx] = (A.dot(np.linalg.solve(sigma_w_inv, A.toarray().T)).diagonal()**0.5).reshape(frame.shape)
            y_model_errs[:, :, idx] = (A.dot(np.linalg.inv(sigma_w_inv).diagonal()**0.5)).reshape(frame.shape)

    if errors:
#        y_model_errs = (y_model_errs.T / np.mean(y_model, axis=(1, 2))).T
#        y_model = (y_model.T / np.mean(y_model, axis=(1, 2))).T
        return y_model, y_model_errs, np.asarray(ws)
#    y_model = (y_model.T / np.mean(y_model, axis=(1, 2))).T
    return y_model, np.asarray(ws)


def fit_transit(t, lcs, lcs_err, orbits, xshift, background, r, t0_val, period_val, r_star, m_star, exptime, sample=True, fit_period=True):

    x, y, yerr = np.hstack(t), np.hstack(lcs), np.hstack(lcs_err)
#    cadence_mask = y > 15
    cadence_mask = np.ones(len(y), bool)
    breaks = np.where(np.diff(x) > 2)[0] + 1
    long_time = [(i - i.mean())/(i.max() - i.min()) for i in np.array_split(x, breaks)]
    orbit_trends = [np.hstack([(t[idx][o] - t[idx][o].min())/(t[idx][o].max() - t[idx][o].min()) - 0.5 for o in orbits[idx][:, np.any(orbits[idx], axis=0)].T]) for idx in range(len(t))]

    X = lambda x: np.vstack([np.atleast_2d(x[idx]).T * np.diag(np.ones(len(x)))[idx] for idx in range(len(x))])

    orbit = X(orbit_trends)
    bkg = X(background)
    xs = X(xshift)
    star = X(long_time)
    ones = X([x1**0 for x1 in xshift])
    hook = X([-np.exp(-300 * (t1 - t1[0])) for t1 in t])

    A = np.hstack([hook, xs, bkg, ones, orbit, orbit**2, star, star**2])

    # Hacky garbage
    A = np.nan_to_num(A)
    A = np.asarray(A)

    with pm.Model() as model:

        # The baseline flux
        norm = pm.Normal('norm', mu=y.mean(), sd=y.std(), shape=len(breaks) + 1)

        normalization = pm.Deterministic('normalization', tt.concatenate([norm[idx] + np.zeros_like(x) for idx, x in enumerate(np.array_split(x, breaks))]))

        # The time of a reference transit for each planet
        t0 = pm.Uniform("t0", lower=t0_val-period_val/2, upper=t0_val+period_val/2, testval=t0_val)
        if fit_period:
            period = pm.Normal('period', mu=period_val, sd=0.0001)
        else:
            period = period_val

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        #u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
        u = pm.Uniform('u', lower=0, upper=1, testval=0.5, shape=1)
        r_star_pm = pm.Normal('r_star', mu=r_star, sd=0.1*r_star)
        r = pm.Normal(
            "r", mu=r, sd=r*0.3)
        ror = pm.Deterministic("ror", r / r_star_pm)
        b = xo.ImpactParameter("b", ror=ror)

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, r_star=r_star_pm, m_star=m_star, b=b)

        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=x, texp=exptime
        )

        light_curve = pm.Deterministic('light_curve', (pm.math.sum(light_curves, axis=-1)) + 1)

        sigma_w_inv = tt.dot(A[cadence_mask].T, A[cadence_mask]/yerr[cadence_mask, None]**2)
        B = tt.dot(A[cadence_mask].T, (y - (light_curve * normalization))[cadence_mask]/yerr[cadence_mask]**2)
        w = tt.slinalg.solve(sigma_w_inv, B)
        noise_model = pm.Deterministic('noise_model', tt.dot(A, w))


        no_limb = pm.Deterministic('no_limb', pm.math.sum(xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(orbit=orbit, r=r, t=x, texp=exptime), axis=-1) + 1)

        full_model = pm.Deterministic('full_model', (light_curve * normalization) + noise_model)
        pm.Normal("obs", mu=full_model, sd=yerr, observed=y)


        map_soln = xo.optimize(start=None, verbose=False)
        if sample:
            trace = xo.sample(
                tune=200, draws=1000, start=map_soln, chains=4, target_accept=0.95
            )
            return map_soln, pm.trace_to_dataframe(trace)
        return map_soln


def get_vsr_slant(visit):

    k = np.abs(visit.X) < -visit.X[0][0][5]
    weights = np.copy(1/visit.error)
    weights[~k] = 0
#    vsr_mean = np.average(visit.data, weights=weights, axis=2)[:, :, None] * np.ones(visit.shape)
#    vsr_mean = (vsr_mean.T/np.mean(vsr_mean, axis=(1, 2))).T

    m = visit.average_lc[:, None, None] * visit.spec_mean
    frames = visit.data / m
    frames_err = visit.error / m
    if hasattr(visit, 'model_lc_no_ld'):
        frames_err = (frames_err/np.average(frames[visit.model_lc_no_ld == 1], weights=1/frames_err[visit.model_lc_no_ld == 1], axis=(0)))
        frames = (frames/np.average(frames[visit.model_lc_no_ld == 1], weights=1/frames_err[visit.model_lc_no_ld == 1], axis=(0)))
    else:
        frames_err = (frames_err/np.average(frames, weights=1/frames_err, axis=(0)))
        frames = (frames/np.average(frames, weights=1/frames_err, axis=(0)))

    X = np.vstack([visit.X[0][0]**0, visit.X[0][0], visit.X[0][0]**2, visit.X[0][0]**3]).T
    model = np.zeros_like(visit.data)
    prior_sigma = np.ones(4) * 10
    prior_mu = np.asarray([1, 0, 0, 0])

    for tdx in range(visit.nt):
        for idx in range(visit.nsp):
            sigma_w_inv = X[k[tdx][idx]].T.dot(X[k[tdx][idx]]/frames_err[tdx][idx][k[tdx][idx], None]**2)
            sigma_w_inv += np.diag(1/prior_sigma**2)
            B = X[k[tdx][idx]].T.dot(frames[tdx][idx][k[tdx][idx]]/frames_err[tdx][idx][k[tdx][idx]]**2)
            B += prior_mu/prior_sigma**2
            model[tdx][idx] = X.dot(np.linalg.solve(sigma_w_inv, B))

    vsr_mean = np.mean(model, axis=2)[:, : ,None] * np.ones(visit.shape)
    vsr_grad = model - vsr_mean
    return vsr_mean, vsr_grad


def _estimate_arclength(centroid_col, centroid_row):
    """Estimate the arclength given column and row centroid positions.
    We use the approximation that the arclength equals
        (row**2 + col**2)**0.5
    For this to work, row and column must be correlated not anticorrelated.
    """
    col = centroid_col - np.nanmin(centroid_col)
    row = centroid_row  - np.nanmin(centroid_row)
    # Force c to be correlated not anticorrelated
    if np.polyfit(col.data, row.data, 1)[0] < 0:
        col = np.nanmax(col) - col
    return (col**2 + row**2)**0.5



def _get_matrices(visit, npoly=2):
    """ Make a design matrix for each channel """
    arclength = _estimate_arclength(visit.xshift, visit.yshift)
    arclength = arclength[:, None, None] * np.ones(visit.shape)

    a = arclength[:, :, 0].ravel()[:, None]
    a -= np.median(a)
    b = visit.Y[:, :, 0].ravel()[:, None]
    c = visit.T[:, :, 0].ravel()[:, None]
    d = (visit.bkg[:, None] * np.ones((visit.nt, visit.nsp))).ravel()[:, None]

    a /= a.std()
    b /= b.std()
    c /= c.std()
    d /= d.std()

    if npoly == 2:
        X = np.hstack([np.ones(visit.nt*visit.nsp)[:, None],
                       a, b, #c, c**2, d
                       (a * b),
                       a**2, b**2, a**2*b, a*b**2, a**2*b**2
                       ])
    elif npoly == 3:
        X = np.hstack([np.ones(visit.nt*visit.nsp)[:, None],
                       a, b, #d,
                       (a * b),
                       a**2, b**2, a**2*b, a*b**2, a**2*b**2,
                       a**3, b**3, a**3*b, a**3*b**2, a*b**3, a**2*b**3, a**3*b**3])
    else:
        raise ValueError("Can not process that npoly yet.")

    prior_sigma = np.ones(X.shape[1]) * 0.1
    prior_mu = np.zeros(X.shape[1])
    prior_mu[0] = 1
    return X, prior_mu, prior_sigma



def _make_dict(obs, npoly=2):
    ''' Make a dictionary of the data from all the visits '''
    d = {k: [] for k in ['Xs', 'Trs', 'prior_mu', 'prior_sigma', 'wavelengths', 'visit_numbers', 'data', 'error', 'times']}
    for idx, visit in enumerate(obs):
        X, prior_mu, prior_sigma = _get_matrices(visit, npoly=npoly)
        d['Xs'].append(X)
        d['prior_mu'].append(prior_mu)
        d['prior_sigma'].append(prior_sigma)
        d['Trs'].append((visit.model_lc_no_ld - 1)[:, None, None] * np.ones(visit.shape))
        d['wavelengths'].append(visit.wavelength)
        d['visit_numbers'].append(np.ones(visit.nwav) * idx)

        m = (visit.vsr_mean + visit.vsr_grad) * visit.average_lc[:, None, None] * visit.spec_mean
        frames = visit.data / m
        frames_err = visit.error / m
        frames_err = (frames_err/np.average(frames[visit.model_lc_no_ld == 1], weights=1/frames_err[visit.model_lc_no_ld == 1], axis=(0)))
        frames = (frames/np.average(frames[visit.model_lc_no_ld == 1], weights=1/frames_err[visit.model_lc_no_ld == 1], axis=(0)))

        # avg = (np.average(frames, axis=2, weights=1/frames_err))
        # std = (np.average((frames - avg[:, :, None])**2, axis=2, weights=1/frames_err))**0.5
        # std /= visit.nwav**0.5
        #
        # bad = (np.abs(avg - 1)/std > 5)[:, :, None] * np.ones(visit.shape, bool)
        #
        # mask = ((frames)/frames_err) > 1
        # mask &= ~visit.cosmic_rays
        mask = (np.abs(visit.Y) < -visit.Y[0][2][0])
        # mask &= ~bad

        frames_err[~mask] = 1e10
        if visit.filters[0] == 'G141':
            frames_err[:, :, visit.wavelength <= 11250] = 1e+10
            frames_err[:, :, visit.wavelength >= 16500] = 1e+10
        else:
            frames_err[:, :, visit.wavelength <= 7800] = 1e+10
            frames_err[:, :, visit.wavelength >= 11300] = 1e+10

        d['data'].append(frames)
        d['error'].append(frames_err)
        d['times'].append(visit.time[:, None, None] * np.ones(visit.shape))
    return d



def fit_transmission_spectrum(obs, wav_grid1, wav_grid2, npoly=2):
    td = np.zeros(len(wav_grid1)) * np.nan
    tde = np.zeros(len(wav_grid1)) * np.nan
    dic = _make_dict(obs, npoly=npoly)
    model = [np.copy(dat1) * np.nan for dat1 in dic['data']]

    ws = np.hstack(dic['wavelengths'])
    vmasks = []
    waxis = np.zeros(len(wav_grid1))
    for idx, w1, w2 in zip(range(len(wav_grid1)), wav_grid1, wav_grid2):
        vmask = (ws >= w1) & (ws < w2)
        vmasks.append(np.array_split(vmask, np.where(np.diff(np.hstack(dic['visit_numbers'])) != 0)[0] + 1))
        waxis[idx] = np.mean(ws[vmask])

    for wdx in tqdm(np.arange(0, len(wav_grid1))):
        X = []
        prior_sigma = []
        prior_mu = []
        d = []
        e = []
        tr = []
        ti = []
        for vdx, vmask in enumerate(vmasks[wdx]):
            nchannels = vmask.sum()
            _ = [X.append(dic['Xs'][vdx]) for i in range(vmask.sum())]
            _ = [prior_sigma.append(dic['prior_sigma'][vdx]) for i in range(vmask.sum())]
            _ = [prior_mu.append(dic['prior_mu'][vdx]) for i in range(vmask.sum())]
            d.append(dic['data'][vdx][:, :, vmask].ravel())
            e.append(dic['error'][vdx][:, :, vmask].ravel())
            tr.append(dic['Trs'][vdx][:, :, vmask].ravel())
            ti.append(dic['times'][vdx][:, :, vmask].ravel())
        if len(X) == 0:
            continue
        Xl = sparse.lil_matrix((np.sum([len(x) for x in X]), np.sum([x.shape[1] for x in X])))
        lastx = 0
        lasty = 0
        for x in X:
            Xl[lastx:lastx+x.shape[0], lasty:lasty+x.shape[1]] = x
            lastx += x.shape[0]
            lasty += x.shape[1]
        d, e, tr, ti = np.hstack(d), np.hstack(e), np.hstack(tr), np.hstack(ti)
        if (e >= 1e10).all():
            continue
        prior_mu = np.hstack(prior_mu)
        prior_sigma = np.hstack(prior_sigma)
        Xl = sparse.hstack([Xl, tr[:, None]], format='csr')
        prior_mu = np.append(prior_mu, 0)
        prior_sigma = np.append(prior_sigma, 0.05)

        sigma_w_inv = Xl.T.dot(Xl/e[:, None]**2)
        sigma_w_inv += np.diag(1/prior_sigma**2)
        B = Xl.T.dot(d/e**2)
        B += prior_mu/prior_sigma**2
        w = np.linalg.solve(sigma_w_inv, B)

#        import pdb;pdb.set_trace()
        werr = np.linalg.inv(np.asarray(sigma_w_inv)).diagonal()**0.5
        td[wdx] = w[-1]
        tde[wdx] = werr[-1]

        last = 0
        m = []
        for x in X:
            m.append(x.dot(w[last:last+x.shape[1]]))
            last += x.shape[1]

        ldx = 0
        for vdx, vmask in enumerate(vmasks[wdx]):
            if model[vdx][:, :, vmask].shape[2] == 0:
                continue
            for kdx, vm in enumerate(np.where(vmask)[0]):
                model[vdx][:, :, vm] = m[ldx + kdx].reshape(model[vdx][:, :, vm].shape)
            ldx += 1 + kdx

    for vdx, visit in enumerate(obs):
        m = ((visit.vsr_mean + visit.vsr_grad) * visit.average_lc[:, None, None] * visit.spec_mean) * model[vdx]
        frames = visit.data / m
        frames_err = visit.error / m
#        m *= np.average(frames[visit.model_lc_no_ld == 1], weights=1/frames_err[visit.model_lc_no_ld == 1], axis=(0))
        visit.partial_model = m

    depth = 1 - obs.model_lc_no_ld.min()


    obs.wav_grid = np.vstack([wav_grid1, wav_grid2]).mean(axis=0)
    obs.transmission_spec = td * depth * 1e6
    obs.transmission_spec_err = tde * depth * 1e6
