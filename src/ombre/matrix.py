"""Matrix stacking helper functions"""
import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from scipy import sparse


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


def vstack(vecs, n, n_dependence=None):
    """Stack n input sparse matrices into a large diagnoal matrix where each component is dependent."""
    mats = []
    for vec in vecs:
        vec_s = sparse.lil_matrix(vec)
        if n_dependence is not None:
            mats.append(
                sparse.hstack([vec_s * n_dependence[idx] for idx in np.arange(n)])
            )
        else:
            mats.append(sparse.hstack([vec_s for idx in np.arange(n)]))
    return sparse.vstack(mats).T


def vstack_independent(mat, n):
    """Stack n input sparse matrices into a large diagnoal matrix where each component is independent."""
    mat_s = sparse.csc_matrix(mat)
    npoints = mat.shape[0] * n
    ncomps = mat.shape[1] * n
    X = sparse.csc_matrix((npoints, ncomps))
    idx = 0
    jdx = 0
    for ndx in range(n):
        X[idx : idx + mat.shape[0], jdx : jdx + mat.shape[1]] = mat_s
        idx = idx + mat.shape[0]
        jdx = jdx + mat.shape[1]
    return X
