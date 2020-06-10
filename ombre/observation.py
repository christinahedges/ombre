"""Class to handle all the book-keeping of an HST observation"""

import numpy as np
from glob import glob
from datetime import datetime
import warnings
from tqdm.notebook import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import logging

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clip
from astroquery.mast import Observations as astropyObs

from .methods import simple_mask, average_vsr, average_spectrum, get_flatfield, calibrate, build_vsr, build_spectrum

from starry.extensions import from_nexsci

log = logging.getLogger('ombre')

class Observation(object):

    def __init__(self, filenames, propid=None, visit=None, teff=6000, name=None, t0=None):
        """HST Observation.

        Parameters
        ----------
        filenames : list of str, or str
            List of filenames of HST WFC3 observations, or path to a directory.
        """
        if isinstance(filenames, str):
            if not filenames.endswith('*'):
                self.filenames = np.asarray(glob(filenames + "*"))
            else:
                self.filenames = np.asarray(glob(filenames))
        else:
            self.filenames = np.asarray(filenames)
        if len(self.filenames) == 0:
            raise ValueError('Must pass >0 `filenames`')
        if not np.all([fname.endswith('_flt.fits') for fname in self.filenames]):
            raise ValueError('`filenames` must all be `_flt.fits` files. e.g. {}'.format(self.filenames[0]))

        self.teff = teff
        self.hdrs = np.asarray([fits.getheader(file) for idx, file in enumerate(self.filenames)])

        if name is None:
            self.name = np.asarray([hdr['TARGNAME'] for hdr in self.hdrs])
        else:
            self.name = name
        if not len(np.unique(self.name)) == 1:
            raise ValueError('Multiple targets in `filenames`.')
        self.name = np.unique(self.name)[0]

        try:
            self.sys = from_nexsci(self.name, limb_darkening=[0, 0])
        except ValueError:
            raise ValueError('{} is not a recognized exoplanet name. Please pass a correct `name`.'.format(self.name))
        if t0 is not None:
            self.sys.secondaries[0].t0 = t0


        self.filters = np.asarray([hdr['FILTER'] for hdr in self.hdrs])
        mask = (self.filters == 'G141') | (self.filters == 'G102')

        self.exptime = np.asarray([hdr['EXPTIME'] for hdr in self.hdrs])
        if not (self.exptime[mask] == np.median(self.exptime[mask])).all():
            warnings.warn('Not all files have the same exposure time. {}/{} files will be discarded.'.format((self.exptime[mask] != np.median(self.exptime[mask])).sum(), len(self.exptime[mask])))
        mask &= self.exptime == np.median(self.exptime)

        self.propid = np.asarray([hdr['PROPOSID'] for hdr in self.hdrs])
        if propid is None:
            if len(np.unique(self.propid)) != 1:
                warnings.warn('Data sets from multiple proposals have been passed ({})'.format(np.unique(self.propid)))
        else:
            mask &= self.propid == propid

        if not mask.any():
            raise ValueError('All cadences are masked.')

        convert_time = lambda hdr:(Time(datetime.strptime('{}-{}'.format(hdr['DATE-OBS'],
                                                              hdr['TIME-OBS']), '%Y-%m-%d-%H:%M:%S')).jd,
								   Time(datetime.strptime('{}'.format(hdr['DATE-OBS']), '%Y-%m-%d')).jd)


        self.time, self.start_date = np.asarray([convert_time(hdr) for hdr in self.hdrs]).T
        s = np.argsort(self.time)
        [setattr(self, key, getattr(self, key)[s])
                for key in ['time', 'start_date', 'propid', 'exptime', 'filters', 'hdrs', 'filenames']]
        mask = mask[s]

        self.nvisits = len(np.unique(np.round(self.start_date, -1)))
        self.visits = np.unique(np.round(self.start_date, -1), return_inverse=True)[1] + 1

        if visit != None:
            if visit > self.nvisits:
                raise ValueError('Only {} visits available.'.format(self.nvisits))
            mask &= self.visits == visit
        self.time, self.start_date, self.visits, self.propid, self.exptime, self.filters =\
            self.time[mask], self.start_date[mask], self.visits[mask], self.propid[mask],\
            self.exptime[mask], self.filters[mask]
        self.postarg1 = np.asarray([hdr['POSTARG1'] for hdr in self.hdrs])[mask]
        self.postarg2 = np.asarray([hdr['POSTARG2'] for hdr in self.hdrs])[mask]
        self.sun_alt = np.asarray([hdr['SUN_ALT'] for hdr in self.hdrs])[mask]
        self.hdrs = self.hdrs[mask]

        self.ra = self.hdrs[0]['RA_TARG']
        self.dec = self.hdrs[0]['DEC_TARG']
        self.nt = np.sum(mask)
        aperture = self.hdrs[0]['APERTURE']
        self.ns = int(''.join([a for a in aperture if a.isnumeric()]))
        self._filenames = np.asarray(self.filenames)[mask]
        self.forward = self.postarg2 > 0

        orbits = np.where(np.append(0, np.diff(self.time)) > 0.015)[0]
        self.orbits = np.asarray([np.in1d(np.arange(self.nt), o) for o in np.array_split(np.arange(self.nt), orbits)])

        self._load_data()
        self._build_masks()
        self._build_data()
        self.sensitivity, self.wavelength = calibrate(self, self.teff)


    def _load_data(self):
        """ Helper function to load in the data """
        self.sci, self.err, self.dq = np.zeros((self.nt, self.ns, self.ns)), np.zeros((self.nt, self.ns, self.ns)), np.zeros((self.nt, self.ns, self.ns), dtype=int)
        self.velocity_aberration = np.zeros(self.nt)
        for jdx, file in enumerate(self._filenames):
            hdulist = fits.open(file)
            self.sci[jdx], self.err[jdx], self.dq[jdx] = np.asarray([hdu.data for hdu in hdulist[1:4]])
            if hdulist[1].header['BUNIT'] == 'ELECTRONS':
                self.sci[jdx] /= hdulist[1].header['SAMPTIME']
                self.err[jdx] /= hdulist[1].header['SAMPTIME']
            self.velocity_aberration[jdx] = hdulist[1].header['VAFACTOR']
            hdulist.close('all')
        qmask = 1 | 2 | 4 | 8 | 16 | 32 | 256
        self.err[(self.dq & qmask) != 0] = 1e10


    def _build_masks(self):
        self.mask, self.spectral, self.spatial = simple_mask(self)


    def _build_data(self):
        self.flat = np.ones((1, *self.sci.shape[1:]))
        for count in [0, 1]:
            # We do this twice to improve the averages and flat field estimate
            self.data = (self.sci/self.flat)[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)]
            self.error = (self.err/self.flat)[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)]
            self.nsp, self.nwav = self.shape[1:]

            self.vsr_mean = average_vsr(self)
            self.spec_mean = average_spectrum(self)
            self.model = self.spec_mean * self.vsr_mean
            self.model /= np.atleast_3d(self.model.mean(axis=(1, 2))).transpose([1, 0, 2])
            self.model_err = np.zeros(self.model.shape)
            if count == 0:
                self.flat = get_flatfield(self, self.model)

        self.error[self.error/self.data > 0.1] = 1e10
        res = (self.data / self.model)
        res = ((res.T) - np.mean(res, axis=(1, 2))).T
        self.cosmic_rays = sigma_clip(res, sigma=8, axis=0).mask
        self.error += self.cosmic_rays * 1e10


        T = (np.atleast_3d(self.time) * np.ones(self.shape).transpose([1, 0, 2])).transpose([1, 0, 2])
        T -= self.time[0]
        self.T = (T/T.max() - 0.5)

        Y, X = np.mgrid[:self.shape[1], :self.shape[2]]
        Y = Y/(self.shape[1] - 1) - 0.5
        X = X/(self.shape[2] - 1) - 0.5

        self.X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.data.shape)
        self.Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(self.data.shape)
        self.xshift = np.average(self.X, weights=self.data/self.error, axis=(1, 2))

    def __len__(self):
        return self.nt

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, s):
        copyself = self.copy()
        [setattr(copyself, key, getattr(self, key)[s])
                for key in ['time', 'start_date', 'propid', 'exptime', 'filters',
                            'hdrs', 'postarg1', 'postarg2', 'visits',
                            'sun_alt', 'velocity_aberration', '_filenames', 'sci',
                             'err', 'dq', 'data', 'error', 'vsr_mean', 'spec_mean',
                              'model', 'cosmic_rays', 'X', 'Y', 'T', 'xshift', 'forward'] if hasattr(copyself, key)]
        copyself.nt = len(copyself.time)
        copyself._build_masks()
        copyself._build_data()
        copyself.sensitivity, copyself.wavelength = calibrate(copyself, self.teff)
        return copyself



    @staticmethod
    def from_MAST(targetname, visit=None, direction=None, **kwargs):
        """Download a target from MAST
        """
        def download_target(targetname, radius='10 arcsec'):
            download_dir = os.path.join(os.path.expanduser('~'), '.ombre-cache')
            if not os.path.isdir(download_dir):
                try:
                    os.mkdir(download_dir)
                # downloads locally if OS error occurs
                except OSError:
                    log.warning('Warning: unable to create {}. '
                                'Downloading MAST files to the current '
                                'working directory instead.'.format(download_dir))
                    download_dir = '.'

            logging.getLogger('astropy').setLevel(log.getEffectiveLevel())
            obsTable = astropyObs.query_criteria(target_name=targetname,
                                                    obs_collection='HST',
                                                    project='HST',
                                                    radius=radius)
            obsTable = obsTable[(obsTable['instrument_name'] == 'WFC3/IR') &
                                (obsTable['dataRights'] == "PUBLIC")]

            fnames = ['{}/mastDownload/'.format(download_dir) +
                        url.replace('mast:', '').replace('product', url.split('/')[-1].split('_')[0])[:-9] +
                         '_flt.fits' for url in obsTable['dataURL']]
            log.info('Found {} files.'.format(len(obsTable)))
            paths = []
            for idx, t in enumerate(tqdm(obsTable, desc='Downloading files', total=len(obsTable))):
                if os.path.isfile(fnames[idx]):
                    paths.append(fnames[idx])
                else:
                    t1 = astropyObs.get_product_list(t)
                    t1 = t1[t1['productSubGroupDescription'] == 'FLT']
                    paths.append(astropyObs.download_products(t1, mrp_only=False, download_dir=download_dir)['Local Path'][0])
            return paths

        paths = np.asarray(download_target(targetname))

        if isinstance(visit, int):
            start_time = np.asarray([Time(datetime.strptime('{}'.format(fits.open(fname)[0].header['DATE-OBS']), '%Y-%m-%d')).jd for fname in paths])
            visits = np.asarray([np.where(np.sort(np.unique(start_time)) == t1)[0][0] + 1 for t1 in start_time])
            mask = visits == visit
            if not mask.any():
                raise ValueError('No data in visit {}'.format(visit))
            paths = paths[mask]

        if isinstance(direction, str):
            directions = np.asarray([fits.open(fname)[0].header['POSTARG2'] for fname in paths])
            if direction == 'forward':
                paths = paths[directions >= 0]
            elif direction == 'backward':
                paths = paths[directions <= 0]
            else:
                raise ValueError("Can not parse direction {}. "
                                    "Choose from `'forward'` or `'backward'`".format(direction))

        return Observation(paths, **kwargs)

    def __repr__(self):
        return '{} (WFC3 Observation)'.format(self.name)

    @property
    def model_lc(self):
        return self.sys.flux(self.time).eval()

    @property
    def average_lc(self):
        lc = np.average(self.data/self.model, weights=self.model/self.error, axis=(1, 2))
        norm = np.median(lc[self.model_lc == 1])
        return lc/norm

    @property
    def average_lc_errors(self):
        draws = np.random.normal(self.data/self.model, self.error/self.model, size=(50, *self.shape))
        return np.asarray([np.average(d, weights=np.nan_to_num(self.model/self.error), axis=(1, 2)) for d in draws]).std(axis=0)

    @property
    def channel_lcs(self):
        m = ((self.vsr_mean > 0.93)).astype(float)
        return np.average((self.data/self.model) * m, weights=(self.model/self.error) * m, axis=(1))

    @property
    def channel_lcs_errors(self):
        m = ((self.vsr_mean > 0.93)).astype(float)
        draws = np.random.normal(self.data/self.model, self.error/self.model, size=(50, *self.shape))
        return np.asarray([np.average(d * m, weights=np.nan_to_num(self.model/self.error) * m, axis=(1)) for d in draws]).std(axis=0)

    @property
    def raw_average_lc(self):
        m = ((self.vsr_mean > 0.93)).astype(float)
        return np.average((self.data/self.spec_mean) * m, weights=(self.spec_mean/self.error) * m, axis=(1, 2))

    @property
    def raw_average_lc_errors(self):
        m = ((self.vsr_mean > 0.93)).astype(float)
        draws = np.random.normal(self.data/self.spec_mean, self.error/self.spec_mean, size=(50, *self.shape))
        return np.asarray([np.average(d * m, weights=np.nan_to_num(self.spec_mean/self.error) * m, axis=(1, 2)) for d in draws]).std(axis=0)

    @property
    def raw_channel_lcs(self):
        return np.average(self.data/self.spec_mean, weights=self.spec_mean/self.error, axis=(1))

    @property
    def raw_channel_lcs_errors(self):
        draws = np.random.normal(self.data/self.spec_mean, self.spec_mean/self.error, size=(50, *self.shape))
        return np.asarray([np.average(d, weights=np.nan_to_num(self.model/self.error), axis=(1)) for d in draws]).std(axis=0)

    @property
    def residuals(self):
        r = (self.data / self.model)
        return (r.T - np.average(r, weights=self.model/self.error, axis=(1, 2))).T

    def plot_average_lc(self, ax=None, errors=False):
        if ax is None:
            _, ax = plt.subplots()

        if errors:
            rlc, rlce = self.raw_average_lc, self.raw_average_lc_errors
            lc, lce = self.average_lc, self.average_lc_errors
            lce /= np.median(lc)
            lc /= np.median(lc)
            rlce /= np.median(rlc)
            rlc /= np.median(rlc)
            ax.errorbar(self.time, rlc, rlce, ls='', c='r', label='Raw Light Curve')
            ax.errorbar(self.time, lc, lce, ls='', c='k', label='Corrected Light Curve')
        else:
            rlc = self.raw_average_lc
            lc = self.average_lc
            lc /= np.median(lc)
            rlc /= np.median(rlc)
            ax.scatter(self.time, rlc, c='r', marker='.', label='Raw Light Curve')
            ax.scatter(self.time, lc, c='k', marker='.', label='Corrected Light Curve')
        ax.set(xlabel='Time from Observation Start [d]', ylabel='Normalized Flux', title='Channel Averaged Light Curve')
        t = np.linspace(self.time[0], self.time[-1], 1000)
        ax.plot(t, self.sys.flux(t).eval(), c='b', label='model')
        ax.legend()
        return ax

    def plot_frame(self, frame=0):
        fig = plt.figure(figsize=(10, 10*self.shape[1]/self.shape[2]))
        ax = plt.subplot2grid((1, 3), (0, 0))
        v = np.percentile(self.data[frame], [1, 99])
        im = ax.imshow(self.data[frame], vmin=v[0], vmax=v[1], cmap='viridis')
        ax.set(xlabel='Wavelength Pixel', title='Data', xticklabels=[], yticklabels=[], ylabel='Spatial Pixel')
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
        cbar.set_label('e$^-$/s')

        ax = plt.subplot2grid((1, 3), (0, 1))
        im = ax.imshow(self.model[frame] * self.data[frame].mean(), vmin=v[0], vmax=v[1], cmap='viridis')
        ax.set(xlabel='Wavelength Pixel', title='Model', xticklabels=[], yticklabels=[])
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
        cbar.set_label('e$^-$/s')

        ax = plt.subplot2grid((1, 3), (0, 2))

        res = self.residuals[frame]
        v = np.percentile(np.abs(res), 84)
        im = ax.imshow(res, vmin=-v, vmax=v, cmap='coolwarm')
        ax.set(xlabel='Wavelength Pixel', title='Residuals', xticklabels=[], yticklabels=[])
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
        cbar.set_label('Residual')
        plt.subplots_adjust(wspace=0.02)
        return fig

    def plot_channel_lcs(self, offset=10, lines=True, residuals=False, **kwargs):
        c = np.nanmedian(self.channel_lcs) *  np.ones(self.nt)
        cmap = kwargs.pop('cmap', plt.get_cmap('coolwarm'))
        fig, ax = plt.subplots(1, 2, figsize=(15, 25), sharey=True)

        if residuals:
            [ax[0].scatter(self.time, self.raw_channel_lcs[:, kdx] + kdx * offset - self.wl,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                        vmin=self.wavelength[0], vmax=self.wavelength[-1],
                        cmap=cmap)
                            for kdx in range(len(self.wavelength))];

            [ax[1].scatter(self.time, self.channel_lcs[:, kdx] + kdx * offset - self.wl,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                         vmin=self.wavelength[0], vmax=self.wavelength[-1],
                         cmap=cmap)
                            for kdx in range(len(self.wavelength))];
            if lines:
                [ax[1].plot(self.time, np.ones(self.nt) * kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(self.channel_lcs[:, kdx]) != 0)];

        else:
            [ax[0].scatter(self.time, self.raw_channel_lcs[:, kdx] + kdx * offset,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                        vmin=self.wavelength[0], vmax=self.wavelength[-1],
                        cmap=cmap)
                            for kdx in range(len(self.wavelength))];

            [ax[1].scatter(self.time, self.channel_lcs[:, kdx] + kdx * offset,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                         vmin=self.wavelength[0], vmax=self.wavelength[-1],
                         cmap=cmap)
                            for kdx in range(len(self.wavelength))];
            if lines:
                [ax[1].plot(self.time, c + kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(self.channel_lcs[:, kdx]) != 0)];
        ax[1].set(xlabel='Time', ylabel='Flux', title='Corrected', yticklabels='')
        ax[0].set(xlabel='Time', ylabel='Flux', title='Raw', yticklabels='')
        plt.subplots_adjust(wspace=0.)
        return fig

    def correct_VSR(self, errors=False):
        if hasattr(self, '_vsr_grad_model'):
            warnings.warn('You have already run `correct_VSR`. No correction will be applied')
            return
        if errors:
            self._vsr_grad_model, self._vsr_grad_model_errs = build_vsr(self)
            self.model_err = np.hypot(self.model_err, self._vsr_grad_model_errs) * self.model/self._vsr_grad_model
            self.model *= self._vsr_grad_model
            self.model_err = (self.model_err.T/np.mean(self.model, axis=(1, 2))).T
            self.model = (self.model.T/np.mean(self.model, axis=(1, 2))).T
        else:
            self._vsr_grad_model = build_vsr(self)
            self.model *= self._vsr_grad_model
            self.model = (self.model.T/np.mean(self.model, axis=(1, 2))).T


    def correct_spectrum_tilts(self, errors=False, npoly=2):
        if hasattr(self, '_spec_grad_model'):
            warnings.warn('You have already run `correct_spectrum_tilts`. No correction will be applied')
            return
        if errors:
            self._spec_grad_model, self._spec_grad_model_errs, self._spec_ws = build_spectrum(self, errors=errors, npoly=npoly)
            self.model_err = np.hypot(self.model_err, self._spec_grad_model_errs) * self.model/self._spec_grad_model
            self.model *= self._spec_grad_model
            self.model_err = (self.model_err.T/np.mean(self.model, axis=(1, 2))).T
            self.model = (self.model.T/np.mean(self.model, axis=(1, 2))).T
        else:
            self._spec_grad_model, self._spec_ws = build_spectrum(self, npoly=npoly)
            self.model *= self._spec_grad_model
            self.model = (self.model.T/np.mean(self.model, axis=(1, 2))).T
