"""Common ombre methods"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import warnings

from astropy.io import fits
from astropy.modeling.blackbody import blackbody_lambda
import astropy.units as u

from scipy.optimize import minimize
from scipy import sparse

import lightkurve as lk

from tqdm.notebook import tqdm

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


def simple_mask(observation, spectral_cut=0.2, spatial_cut=0.2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = np.atleast_3d(observation.sci.mean(axis=0) > np.nanpercentile(observation.sci, 50)).transpose([2, 0, 1])

        data = observation.sci/mask
        data[~np.isfinite(data)] = np.nan

        spectral = np.nanmean(data, axis=(0, 1))
        spatial = np.nanmean(data, axis=(0, 2))

        spectral = spectral > np.nanmax(spectral) * spectral_cut
        spatial = spatial > np.nanmax(spatial) * spatial_cut

        mask = (np.atleast_3d(np.atleast_2d(spectral) * np.atleast_2d(spatial).T)).transpose([2, 0, 1])
    return mask, np.atleast_3d(np.atleast_2d(spectral)).transpose([2, 0, 1]), np.atleast_3d(np.atleast_2d(spatial.T)).transpose([2, 1, 0])


def average_vsr(obs):
    avg = np.atleast_3d(np.average(obs.data, weights=1/obs.error, axis=1)).transpose([0, 2, 1])
    vsr_mean = np.asarray([np.average(obs.data[idx]/avg[idx], weights=avg[idx]/obs.error[idx], axis=1) for idx in range(obs.nt)])
    return np.atleast_3d(vsr_mean) * np.ones(obs.shape)


def average_spectrum(obs):
    if hasattr(obs, 'vsr_mean'):
        avg = np.atleast_3d(obs.vsr_mean)
    else:
        avg = np.atleast_3d(np.atleast_3d(np.average(obs.data, weights=1/obs.error, axis=2)).transpose([0, 2, 1]))
    spec_mean = np.average(obs.data/avg, weights=avg/obs.error, axis=(0, 1))
    return (np.atleast_3d(spec_mean) * np.ones(obs.shape).transpose([0, 2, 1])).transpose([0, 2, 1])


def get_flatfield(obs, model):
    ''' Get flat field'''
    g141_flatfile = CALIPATH+'WFC3.IR.G141.flat.2.fits'
    ff_hdu = fits.open(g141_flatfile)

    a1, a2 = np.where(obs.spatial[0, :, 0])[0][0] , np.where(obs.spatial[0, :, 0])[0][-1] + 1
    b1, b2 = np.where(obs.spectral[0, 0, :])[0][0] , np.where(obs.spectral[0, 0, :])[0][-1] + 1

    # Some handy shapes
    nt, nrow, ncol = obs.shape

    norm = np.nansum(obs.sci, axis=(1, 2))
    norm /= np.median(norm)
    norm = np.atleast_3d(norm).transpose([1, 0, 2])

    f = np.asarray([ff_hdu[i].data for i in range(4)])
    f = f[:, 1014//2 - obs.ns//2 : 1014//2 + obs.ns//2, 1014//2 - obs.ns//2 : 1014//2 + obs.ns//2]

    f = np.vstack([f, np.ones((1, f.shape[1], f.shape[2]))])
    X = f[:, a1:a2, :][:, :, b1:b2][:, np.ones((nrow, ncol), bool)].T

    avg = np.nanmean((obs.data/norm/model), axis=0)[np.ones((nrow, ncol), bool)]
    avg_err = ((1/nt)*np.nansum(((obs.error/norm/model))**2, axis=0)**0.5)[np.ones((nrow, ncol), bool)]

    sigma_w_inv = np.dot(X.T, X/avg_err[:, None]**2)
    B = np.dot(X.T, (avg/avg_err**2)[:, None])
    w = np.linalg.solve(sigma_w_inv, B)[:, 0]

    X = f[:, np.ones((obs.ns, obs.ns), bool)].T
    flat = np.dot(X, w).reshape((obs.ns, obs.ns))
    flat /= np.median(flat)
    flat[(obs.dq[0] & 256) != 0] = 1
    flat = np.atleast_3d(flat).transpose([2, 0, 1])
    return flat


def calibrate(obs, teff, kdx=0):
    g141_sensfile = CALIPATH+'WFC3.IR.G141.1st.sens.2.fits'
    hdu = fits.open(g141_sensfile)
    sens_raw = hdu[1].data['SENSITIVITY']
    wav = hdu[1].data['WAVELENGTH'] * u.angstrom
    bb = blackbody_lambda(wav, teff)
    bb /= np.trapz(bb, wav)
    sens = sens_raw * bb.value
    sens /= np.median(sens)
    dw = wav.value[10 + np.argmin(np.gradient(sens[10:-10]))] - wav.value[10 + np.argmax(np.gradient(sens[10:-10]))]
    w_mean = np.mean(wav.value)

    def func(params, return_model=False):
        model = np.interp(np.arange(0, obs.shape[2]), (wav.value - w_mean)/dw *  params[0] + params[1], sens)
        if return_model:
            return model
        return np.sum((obs.spec_mean[0, 0] - model)**2)

    na = 200
    nb = 201
    a = np.linspace(100, 150, na)
    b = np.linspace(40, 100, nb)
    chi = np.zeros((na, nb))
    for idx, a1 in enumerate(a):
        for jdx, b1 in enumerate(b):
            chi[idx, jdx] = np.sum((obs.spec_mean[0, 0] - func([a1, b1], return_model=True))**2)

    l = np.unravel_index(np.argmin(chi), chi.shape)
    params = [a[l[0]], b[l[1]]]
    wavelength = ((np.arange(obs.shape[2]) - params[1]) / params[0]) * dw + w_mean
    sensitivity = np.interp(np.arange(0, obs.shape[2]), (wav.value - w_mean)/dw *  params[0] + params[1], sens_raw)

    wavelength = ((np.arange(obs.shape[2]) - params[1]) / params[0]) * dw + w_mean
    sensitivity = np.interp(np.arange(0, obs.shape[2]), (wav.value - w_mean)/dw *  params[0] + params[1], sens_raw)

    return sensitivity, wavelength



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

    knots = np.linspace(-0.5, 0.5, int(obs.shape[1]))
    spline = lk.designmatrix.create_sparse_spline_matrix(obs.Y[0].ravel(), knots=knots)
    A = sparse.hstack([sparse.csr_matrix(np.ones(np.product(obs.shape[1:]))).T,
                        sparse.csr_matrix(obs.X[0].ravel()).T,
                        spline.X,
                        spline.X.T.multiply(obs.X[0].ravel()).T,
                        spline.X.T.multiply(obs.X[0].ravel()**2).T], format='csr')


    dm = lk.designmatrix.SparseDesignMatrix(A)
    vsr_grad_model = np.zeros(obs.shape)
    if errors:
        vsr_grad_model_errs = np.zeros(obs.shape)
    ws = []
    prior_sigma = np.ones(A.shape[1]) * 100
    prior_mu = np.zeros(A.shape[1])
    prior_mu[0] = 1
    prior_sigma[1] = 0.05

    for idx in tqdm(range(obs.shape[0]), desc='Building Detailed VSR'):
        y = frames[idx].ravel()
        ye = frames_err[idx].ravel()

        # Linear algebra to find the best fitting shifts
        sigma_w_inv = A.T.dot(A/ye[:, None]**2)
        sigma_w_inv += np.diag(1. / prior_sigma**2)
        B = A.T.dot((y/ye**2))
        B += (prior_mu / prior_sigma**2)
        w = np.linalg.solve(sigma_w_inv, B)
        ws.append(w)
        a = A.dot(w).reshape(frames[0].shape)
        vsr_grad_model[idx] = a/a.mean()
        if errors:
            e = (A.dot(np.linalg.solve(sigma_w_inv, A.toarray().T)).diagonal()**0.5).reshape(frames[0].shape)
            vsr_grad_model_errs[idx] = e/e.mean()
    if errors:
        return vsr_grad_model, vsr_grad_model_errs
    return vsr_grad_model



def build_spectrum(obs, errors=False):

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
        wlc = sparse.csr_matrix((np.atleast_2d(obs.average_lc).T * np.ones(obs.shape[:2])).ravel()).T

        At = sparse.hstack([ones, wlc, t, t.multiply(t), t.multiply(t).multiply(t)])
        At2 = sparse.hstack([ones, t2, t2.multiply(t2), t2.multiply(t2).multiply(t2)])
        Ay = sparse.hstack([ones, y, y.multiply(y), y.multiply(y).multiply(y)])

        A = sparse.hstack([(At.T.multiply(A)).T for A in Ay.T])
        A1 = sparse.hstack([(At2[:, 1:].T.multiply(A)).T for A in Ay.T])
        A = sparse.hstack([A, A1])

        prior_mu = np.zeros(A.shape[1])
        prior_sigma = np.ones(A.shape[1]) * 0.5

        A = sparse.hstack([A, rows], format='csr')

        prior_mu = np.hstack([prior_mu, np.ones(rows.shape[1])])
        prior_sigma = np.hstack([prior_sigma, np.ones(rows.shape[1]) * 1])

        return A, prior_mu, prior_sigma

    frames = obs.data / (obs.model)
    frames_err = obs.error  / (obs.model)

    A, prior_mu, prior_sigma = _make_A(obs)

    y_model = np.zeros(obs.shape)
    if errors:
        y_model_errs = np.zeros(obs.shape)

    ws = []
    for idx in tqdm(range(obs.shape[2]), desc='Building Spectrum Model'):
        frame = frames[:, :, idx]
        frame_err = frames_err[:, :, idx]

        y = (frame).ravel()
        ye = (frame_err).ravel()

        # Linear algebra to find the best fitting shifts
        sigma_w_inv = A.T.dot(A/ye[:, None]**2)
        sigma_w_inv += np.diag(1. / prior_sigma**2)
        B = A.T.dot(y/ye**2)
        B += (prior_mu / prior_sigma**2)
        w = np.linalg.solve(sigma_w_inv, B)
        ws.append(w)
        y_model[:, :, idx] = A.dot(w).reshape(frame.shape)
        if errors:
            y_model_errs[:, :, idx] = (A.dot(np.linalg.solve(sigma_w_inv, A.toarray().T)).diagonal()**0.5).reshape(frame.shape)

    if errors:
        y_model_errs = (y_model_errs.T / np.mean(y_model, axis=(1, 2))).T
        y_model = (y_model.T / np.mean(y_model, axis=(1, 2))).T
        return y_model, y_model_errs, np.asarray(ws)
    y_model = (y_model.T / np.mean(y_model, axis=(1, 2))).T
    return y_model, np.asarray(ws)
