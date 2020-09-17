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

from scipy.optimize import minimize
from scipy import sparse

import lightkurve as lk

from fbpca import pca

from tqdm.notebook import tqdm

from scipy.stats import pearsonr

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt

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


def simple_mask(sci, spectral_cut=0.3, spatial_cut=0.3):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = np.atleast_3d(sci.mean(axis=0) > np.nanpercentile(sci, 50)).transpose([2, 0, 1])
        #return mask
        data = sci/mask
        data[~np.isfinite(data)] = np.nan

        spectral = np.nanmean(data, axis=(0, 1))
        spatial = np.nanmean(data, axis=(0, 2))

        spectral = spectral > np.nanmax(spectral) * spectral_cut
        spatial = spatial > np.nanmax(spatial) * spatial_cut

        mask = (np.atleast_3d(np.atleast_2d(spectral) * np.atleast_2d(spatial).T)).transpose([2, 0, 1])

        # Cut out the zero phase dispersion!
        # Choose the widest block as the true dispersion...
        secs = mask.any(axis=0).astype(int)
        len_secs = np.diff(np.where(np.gradient(secs) != 0)[0][::2] + 1)[::2]
        if len(len_secs) > 1:
            a = (np.where(np.gradient(secs) != 0)[0][::2] + 1)[np.argmax(len_secs)*2:np.argmax(len_secs)*2+2]
            b = ((np.in1d(np.arange(len(mask)), np.arange(a[0] - 20, a[1] + 20)))[:, None] * np.ones(len(mask))).T
            mask *= b[None, :, :].astype(bool)

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
    sensitivity = np.interp(wavelength, wav, sens_raw)
    return sensitivity, wavelength


def _build_X(obs, frames, frames_err, spectrum=True, transit=True, spectrum_offset=True, vsr=True, bkg=True, npoly=3):
    """ Build a design matrix for all the data in three dimensions. """

    # Build VSR model
    # Find principle component in spatial dimension
    if vsr:
        As = []
        for tdx in range(obs.nt):
            # Note we build 2 principle components, and only take the first one.
            A, _, _ = pca(frames[tdx], k=np.min([obs.nsp//10, 15]), n_iter=10)
            #A = A[:, 0, None]

            A2 = []
            for idx in range(A.shape[1]):
                A1 = np.hstack([np.atleast_2d(np.ones(obs.nsp)).T, A[:, idx, None]])
                A1 = np.vstack([(np.atleast_2d(a).T * np.ones(frames[0].shape)).ravel() for a in A1.T]).T
                A2.append(A1)
            A = np.hstack(A2)
#            import pdb;pdb.set_trace()
            # Add tilt
#            A = np.hstack([A,
#                         A * np.atleast_2d(obs.X[0].ravel()).T,
#                        ])

            As.append(A)
        As = np.asarray(As)
    else:
        As = np.ones((obs.nt, obs.nwav*obs.nsp, 1))
    X1 = sparse.lil_matrix((np.product(obs.shape), (As.shape[2] * obs.nt)))
    for tdx in range(obs.nt):
        X1[tdx * obs.nsp * obs.nwav : (tdx + 1) * obs.nsp * obs.nwav, tdx * As.shape[2] : (tdx + 1) * As.shape[2]] = As[tdx]

    prior_mu = [0] * As.shape[2]
    prior_sigma = [0.05] * As.shape[2]
    prior_mu = np.hstack(prior_mu * obs.nt).astype(float)
    prior_sigma = np.hstack(prior_sigma * obs.nt).astype(float)

    X = X1

    if spectrum:

        # Build spectral dimension
        def _make_A(obs, npoly):

            xshift = obs.xshift
            xshift -= np.mean(xshift)
            xshift /= (np.max(xshift) - np.min(xshift))
            xshift = np.atleast_3d(xshift).transpose([1, 0, 2]) * np.ones(obs.shape)
            t2 = sparse.csr_matrix(xshift[:, :, 0].ravel()).T

            y, t = sparse.csr_matrix(obs.Y[:, :, 0].ravel()).T, sparse.csr_matrix(obs.T[:, :, 0].ravel()).T

            ones = sparse.csr_matrix(np.ones(y.shape[0])).T
            #delta_depth = sparse.csr_matrix((np.atleast_2d((obs.model_lc != 1).astype(float)).T * np.ones(obs.shape[:2])).ravel()).T

            def poly(x, npoly):
                mul = lambda x: x.multiply(x)
                r = sparse.csr_matrix(np.ones(x.shape[0])).T
                r = sparse.hstack([r, x], format='csr')
                for i in range(npoly - 2):
                    r = sparse.hstack([r, mul(r[:, -1])], format='csr')
                return r

            At = poly(t, npoly)
            At2 = poly(t2, npoly)
            Ay = poly(y, npoly)

            A = sparse.hstack([(At.T.multiply(A)).T for A in Ay.T], format='csr')
            A1 = sparse.hstack([(At2[:, 1:].T.multiply(A)).T for A in Ay.T], format='csr')

            A = sparse.hstack([A, A1])
            return A


        A = _make_A(obs, npoly)
        X2 = sparse.lil_matrix((np.product(obs.shape), (A.shape[1] * obs.nwav)))
        x = np.unique(obs.X)
        for idx in range(obs.nwav):
            X2[obs.X.ravel() == x[idx], A.shape[1] * idx : A.shape[1] * (idx + 1)] = A
        prior_mu = np.hstack([prior_mu, np.zeros(X2.shape[1])])
        prior_sigma = np.hstack([prior_sigma, np.ones(X2.shape[1]) * 0.05])

        X = sparse.hstack([X, X2], format='csr')


    if bkg:
        bkg0 = (np.ones(obs.shape).T * obs.bkg).T
        bkg1 = (obs.spec_mean.T * obs.bkg).T
        bkg2 = (obs.spec_mean.T**2 * obs.bkg).T
        bkg3 = (obs.spec_mean.T**3 * obs.bkg).T
        X_bkg = sparse.csr_matrix(np.vstack([bkg0.ravel(), bkg1.ravel(), bkg2.ravel(), bkg3.ravel()]).T)
        X = sparse.hstack([X, X_bkg], format='csr')
        prior_sigma = np.hstack([prior_sigma, [0.05, 0.05, 0.05, 0.05]])
        prior_mu = np.hstack([prior_mu, [0, 0, 0, 0]])



    if spectrum_offset:
        offset = sparse.hstack([sparse.csr_matrix((obs.spec_mean == s).ravel().astype(float)).T
                                    for s in obs.spec_mean[0, 0]])

        prior_mu = np.hstack([prior_mu, obs.spec_mean[0, 0]])
        prior_sigma = np.hstack([prior_sigma, np.ones(obs.nwav) * 0.05])
        X = sparse.hstack([X, offset], format='csr')


    if transit:
       A = sparse.csr_matrix((np.atleast_2d((obs.model_lc - obs.model_lc_no_ld).astype(float)).T * np.ones(obs.shape[:2])).ravel()).T
       X3 = sparse.lil_matrix((np.product(obs.shape), (A.shape[1] * obs.nwav)))
       x = np.unique(obs.X)
       for idx in range(obs.nwav):
           X3[obs.X.ravel() == x[idx], A.shape[1] * idx : A.shape[1] * (idx + 1)] = A
       prior_mu = np.hstack([prior_mu, np.zeros(X3.shape[1])])
       prior_sigma = np.hstack([prior_sigma, np.ones(X3.shape[1]) * 0.05])

       X = sparse.hstack([X, X3], format='csr')


       A = sparse.csr_matrix((np.atleast_2d((obs.model_lc_no_ld - 1).astype(float)).T * np.ones(obs.shape[:2])).ravel()).T
       X3 = sparse.lil_matrix((np.product(obs.shape), (A.shape[1] * obs.nwav)))
       x = np.unique(obs.X)
       for idx in range(obs.nwav):
           X3[obs.X.ravel() == x[idx], A.shape[1] * idx : A.shape[1] * (idx + 1)] = A
       prior_mu = np.hstack([prior_mu, np.zeros(X3.shape[1])])
       prior_sigma = np.hstack([prior_sigma, np.ones(X3.shape[1]) * 0.05])

       X = sparse.hstack([X, X3], format='csr')

    return X, prior_mu, prior_sigma


def fit_data(obs):

    d = ma.masked_array(obs.data, mask=(obs.error/obs.data > 0.1))
    e = ma.masked_array(obs.error, mask=(obs.error/obs.data > 0.1))

    wlc = np.average(d, weights=1/e, axis=(1, 2))
    m = (obs.spec_mean * obs.vsr_mean * np.atleast_3d(wlc).transpose([1, 0, 2])).data

    frames = obs.data / m
    frames_err = obs.error  / m

    frames_err = (frames_err.T/np.average(frames, weights=1/obs.error, axis=(1, 2))).T
    frames = (frames.T/np.average(frames, weights=1/obs.error, axis=(1, 2))).T
    frames[(frames_err/frames) > 0.1] = 1

    X, prior_mu, prior_sigma = _build_X(obs, frames, frames_err)


    sigma_f_inv = sparse.csr_matrix(1/frames_err.ravel()**2)
    sigma_w_inv = X.T.dot(X.multiply(sigma_f_inv.T)).toarray()
    sigma_w_inv += np.diag(1. / prior_sigma**2)
    B = X.T.dot((frames/frames_err**2).ravel())
    B += (prior_mu / prior_sigma**2)
    w = np.linalg.solve(sigma_w_inv, B)
#    model1 = X.dot(w).reshape(frames.shape)

#    import pdb;pdb.set_trace()
    model = X[:, :-obs.nwav].dot(w[:-obs.nwav]).reshape(frames.shape)

#    import pdb;pdb.set_trace()

    m = obs.spec_mean * obs.vsr_mean * model

    d = ma.masked_array(obs.data/m, mask=(obs.error/obs.data > 0.1))
    e = ma.masked_array(obs.error/m, mask=(obs.error/obs.data > 0.1))

    ff = np.average(d, weights=1/e, axis=0)/np.average(d, weights=1/e)
    ff.data[ff.mask] = 1

    m *= ff.data

    d = ma.masked_array(obs.data/m, mask=(obs.error/obs.data > 0.1))
    e = ma.masked_array(obs.error/m, mask=(obs.error/obs.data > 0.1))

    stats = sigma_clipped_stats(d, sigma=5)
    cmr = ((obs.data/m - stats[1]) > stats[2]*5)

    d = ma.masked_array(obs.data/m, mask=(obs.error/obs.data > 0.1) | cmr)
    e = ma.masked_array(obs.error/m, mask=(obs.error/obs.data > 0.1) | cmr)


    d /= (np.atleast_3d(np.average(d, weights=1/e, axis=(1, 2)))).transpose([1, 0, 2])

    br = d.std(axis=(0, 2))
    stats = sigma_clipped_stats(br, sigma=5)
    b = (br - stats[1]) > stats[2]*5
    bad_rows = (np.atleast_3d(b.data) * np.ones(obs.shape)).astype(bool)

    for idx in range(obs.nt):
        sigma = np.std(d[idx], axis=1)
        stats = sigma_clipped_stats(sigma, sigma=5)
        bad_rows[idx][(sigma - stats[1]) > stats[2] * 5] |= True

    corr = np.zeros(obs.shape)
    for tdx in range(obs.nt):
        corr[tdx] = np.atleast_2d([np.nan if d[tdx][idx].mask.all() else np.log10(pearsonr(obs.X[tdx][idx][7:-7], d[tdx][idx][7:-7])[1]) for idx in range(obs.nsp)]).T
    bad_rows |= corr < -3

    edge = np.atleast_3d(obs.vsr_mean.mean(axis=0)).transpose([2, 0, 1]) * np.ones(obs.shape) < 0.98

    model = ma.masked_array(m, mask=(obs.error/obs.data > 0.1) | cmr | bad_rows | edge)
    model = (model.T/model.mean(axis=(1, 2))).T

    return model

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


#     p = np.asarray([pca(frames[idx][:, obs.spec_mean[idx][0] > 0.9], k=2, n_iter=20)[0] for idx in tqdm(range(obs.nt))])
#     p1 = (np.atleast_3d(p[:, :, 0]) * np.ones(frames.shape))
# #    p2 = (np.atleast_3d(p[:, :, 1]) * np.ones(frames.shape))
#
#     A = np.vstack([np.ones(np.product(frames.shape)),
#                             (p1 * obs.X).ravel(), (p1 * obs.X**2).ravel(), (p1 * obs.Y).ravel(), (p1 * obs.Y**2).ravel(), (p1 * obs.Y * obs.X).ravel()]).T
#     mask = ((obs.spec_mean > 0.7)).ravel()
#     sigma_w_inv = A[mask].T.dot(A[mask]/frames_err.ravel()[mask, None]**2)
#     B = A[mask].T.dot(frames.ravel()[mask]/frames_err.ravel()[mask]**2)
#     w = np.linalg.solve(sigma_w_inv, B)
#     print(w)
#     model = A.dot(w).reshape(frames.shape)
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



def build_spectrum(obs, errors=False, npoly=2):

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

        At = poly(t, npoly)
        At2 = poly(t2, npoly)
        Ay = poly(y, npoly)
#        At = sparse.hstack([ones, t, t.multiply(t), t.multiply(t).multiply(t)])
#        At2 = sparse.hstack([ones, t2, t2.multiply(t2), t2.multiply(t2).multiply(t2)])
#        Ay = sparse.hstack([ones, y, y.multiply(y), y.multiply(y).multiply(y)])


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

#
# def fit_transit(obs, sample=True):
#     x, y, yerr = obs.time, obs.average_lc, obs.average_lc_errors
#
#     t0_val = obs.sys.secondaries[0].t0.eval()
#     period = obs.sys.secondaries[0].porb.eval()
#     o = np.hstack([(obs.time[o] - obs.time[o].min())/(obs.time[o].max() - obs.time[o].min()) - 0.5 for o in obs.orbits.T])
#     exptime = np.median(obs.exptime) / (3600 * 24)
#
#     with pm.Model() as model:
#
#         # The baseline flux
#
#         norm = pm.Normal("norm", mu=y.mean(), sd=y.std())
#
#         r_star = obs.sys.primary.r.eval()
#         m_star = obs.sys.primary.m.eval()
#
#         # The time of a reference transit for each planet
#         t0 = pm.Uniform("t0", lower=t0_val-period/2, upper=t0_val+period/2, testval=t0_val)
#
#         # The Kipping (2013) parameterization for quadratic limb darkening paramters
#         u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
#
#         r = pm.Normal(
#             "r", mu=obs.sys.secondaries[0].r.eval(), sd=obs.sys.secondaries[0].r.eval()*0.3)
#         ror = pm.Deterministic("ror", r / r_star)
#         b = xo.ImpactParameter("b", ror=ror)
#
#         # Set up a Keplerian orbit for the planets
#         orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, r_star=r_star, m_star=m_star, b=b)
#
#         # Compute the model light curve using starry
#         light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
#             orbit=orbit, r=r, t=x, texp=exptime
#         )
#
#         light_curve = pm.Deterministic('light_curve', (pm.math.sum(light_curves, axis=-1)))
#         if np.asarray(obs.forward).all() | np.asarray(~obs.forward).all():
#             w = pm.Normal('w', mu=0, sd=10, testval=0, shape=3)
#             noise_model = pm.Deterministic('noise_model', w[0] * obs.xshift +
#                                                           w[1] * o +
#                                                           w[2] * obs.bkg)
#         else:
#             fw = obs.forward.astype(float)
#             bw = (~obs.forward).astype(float)
#
#             w = pm.Normal('w', mu=0, sd=10, testval=0, shape=6)
#             noise_model = pm.Deterministic('noise_model', fw * w[0] * obs.xshift +
#                                                           fw * w[1] * o +
#                                                           fw * w[2] * obs.bkg +
#                                                           bw * w[3] * obs.xshift +
#                                                           bw * w[4] * o +
#                                                           bw * w[5] * obs.bkg
#                                                         )
#
#             # Compute the model light curve using starry
#         no_limb = pm.Deterministic('no_limb', pm.math.sum(xo.LimbDarkLightCurve(np.asarray([0, 0])).get_light_curve(orbit=orbit, r=r, t=x, texp=exptime), axis=-1))
#
#
#         if np.asarray(obs.forward).all() | np.asarray(~obs.forward).all():
#             s = pm.Normal('s', mu=np.asarray([1, 0, 0]), sd=np.asarray([0.1, 0.1, 0.1]), shape=3)
#             star_model = pm.Deterministic('star_model', s[0] + s[1] * (x - x.mean()) + s[2] * (x - x.mean()))
#         else:
#             s = pm.Normal('s', mu=np.asarray([1, 1, 0, 0]), sd=np.asarray([0.1, 0.1, 0.1, 0.1]), shape=4)
#             fw = obs.forward.astype(float)
#             bw = (~obs.forward).astype(float)
#             star_model = pm.Deterministic('star_model', s[0] * bw + s[1] * fw +
#                                                         s[2] * (x - x.mean()) +
#                                                         s[3] * (x - x.mean())**2)
#
#         A = pm.Uniform('A', lower=0, upper=0.3, testval=1e-5)
#         v = pm.Uniform('v', lower=-300, upper=-50, testval=-100)
#         hook_model = pm.Deterministic('hook_model', A * -np.exp(v * (x - x[0])))
#
#
#         full_model = pm.Deterministic('full_model', (light_curve + noise_model + star_model + hook_model) * norm)
#
#         pm.Normal("obs", mu=full_model, sd=yerr, observed=y)
#         map_soln = xo.optimize(start=None, verbose=False)
#         if sample:
#             trace = xo.sample(
#                 tune=500, draws=1000, start=map_soln, chains=4, target_accept=0.95
#             )
#             return map_soln, pm.trace_to_dataframe(trace)
#         return map_soln


def fit_transit(t, lcs, lcs_err, orbits, xshift, background, r, t0_val, period_val, r_star, m_star, exptime, sample=True):

    x, y, yerr = np.hstack(t), np.hstack(lcs), np.hstack(lcs_err)
    cadence_mask = y > 15
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

    with pm.Model() as model:

        # The baseline flux
        norm = pm.Normal('norm', mu=y.mean(), sd=y.std(), shape=len(breaks) + 1)

        normalization = pm.Deterministic('normalization', tt.concatenate([norm[idx] + np.zeros_like(x) for idx, x in enumerate(np.array_split(x, breaks))]))

        # The time of a reference transit for each planet
        t0 = pm.Uniform("t0", lower=t0_val-period_val/2, upper=t0_val+period_val/2, testval=t0_val)
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


def fit_transmission_spectrum(obs):
    ts = []
    ts_err = []
    models = []
    partial_models = []
    ws = []
    sigma_ws = []

    for visit in obs.visits:
        if (visit.model_lc == 1).all():
            ts.append(None)
            ts_err.append(None)
            models.append(np.ones(visit.shape) * np.nan)
            continue

        in_transit = (visit.model_lc < visit.model_lc.min() + (1 - visit.model_lc.min()) * 0.2)
        oot_transit = (visit.model_lc == 1)
        wlc = visit.average_lc
        m = (visit.spec_mean * visit.vsr_mean * np.atleast_3d(wlc).transpose([1, 0, 2])).data
        frames = visit.data / m
        frames_err = visit.error  / m

        frames_err = (frames_err.T/np.average(frames, weights=1/visit.error, axis=(1, 2))).T
        frames = (frames.T/np.average(frames, weights=1/visit.error, axis=(1, 2))).T
        frames[(frames_err/frames) > 0.1] = 1
        frames[np.abs(frames - 1) > 0.3] = 1

        X1, prior_mu1, prior_sigma1 = _build_X(visit, frames, frames_err, transit=False)

        m = (visit.vsr_mean * np.atleast_3d(wlc).transpose([1, 0, 2])).data

        frames = visit.data / m
        frames_err = visit.error / m

        X_tr = sparse.lil_matrix((np.product(visit.shape), (visit.nwav)))

        model_lc = visit.model_lc - 1
        model_lc_no_ld = visit.model_lc_no_ld - 1

#        ld = model_lc - model_lc_no_ld
#        ld = ((np.ones(visit.shape).T * ld).T).ravel()
#        ld = sparse.hstack([sparse.csr_matrix(ld).T, sparse.csr_matrix(ld * visit.X.ravel()).T ], format='csr')
        A = sparse.csr_matrix((np.atleast_2d((model_lc - model_lc_no_ld).astype(float)).T * np.ones(visit.shape[:2])).ravel()).T
        x = np.unique(visit.X)
        ld = X_tr.multiply(0).copy()
        for jdx in range(visit.nwav):
            ld[visit.X.ravel() == x[jdx], A.shape[1] * jdx : A.shape[1] * (jdx + 1)] = A

        A = sparse.csr_matrix((np.atleast_2d((model_lc_no_ld).astype(float)).T * np.ones(visit.shape[:2])).ravel()).T
        x = np.unique(visit.X)
        X_tr = X_tr.multiply(0)
        for jdx in range(visit.nwav):
            X_tr[visit.X.ravel() == x[jdx], A.shape[1] * jdx : A.shape[1] * (jdx + 1)] = A

        X_tr2 = sparse.hstack([ld, X_tr], format='csr')


        X = sparse.hstack([X1, X_tr2], format='csr')
        prior_mu = np.hstack([prior_mu1, [0] * 2 * visit.nwav])
        prior_sigma = np.hstack([prior_sigma1, [0.05] * 2 * visit.nwav])

        sigma_f_inv = sparse.csr_matrix(1/frames_err.ravel()**2)
        sigma_w_inv = X.T.dot(X.multiply(sigma_f_inv.T)).toarray()
        sigma_w_inv += np.diag(1. / prior_sigma**2)
        B = X.T.dot((frames/frames_err**2).ravel())
        B += (prior_mu / prior_sigma**2)
        w = np.linalg.solve(sigma_w_inv, B)
        model = (X.dot(w)).reshape(visit.shape)
        #sigma_w = np.linalg.solve(sigma_w_inv, np.eye(w.shape[0]))
        sigma_w = np.linalg.inv(sigma_w_inv)

        ff = np.average((frames-model)[oot_transit], weights=1/frames_err[oot_transit], axis=0)

        B = X.T.dot(((frames - ff)/frames_err**2).ravel())
        B += (prior_mu / prior_sigma**2)
        w = np.linalg.solve(sigma_w_inv, B)
        model = (X.dot(w)).reshape(visit.shape)
        partial_model = (X1.dot(w[:X1.shape[1]])).reshape(visit.shape)
        partial_model -= np.mean(partial_model)
        partial_model += np.mean(model)
        model += ff
        partial_model += ff

        depth = -model_lc.min()

        ts.append(depth * 1e6 * w[-visit.nwav:])
        ts_err.append(depth * 1e6 * (np.diag(sigma_w)**0.5)[-visit.nwav:])
        models.append(model)
        partial_models.append(partial_model)
        ws.append(w)
        sigma_ws.append(np.diag(sigma_w)**0.5)

    return ts, ts_err, models, partial_models, ws, sigma_ws
