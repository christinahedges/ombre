"""Class to handle all the book-keeping of an HST observation"""

import numpy as np
from numpy import ma
from glob import glob
from datetime import datetime
import warnings
from tqdm.notebook import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import logging
from scipy import sparse
import pickle

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.convolution import convolve, Box2DKernel

from astroquery.mast import Observations as astropyObs

from .methods import *

from starry.extensions import from_nexsci

log = logging.getLogger('ombre')

class Visit(object):

    def __init__(self, observation, visit_number, direction='forward', cadence_mask=None):
        if cadence_mask is None:
            cadence_mask = np.ones(observation.nt, bool)

        self.visit_number = visit_number
        self.name = observation.name

        s = np.copy(cadence_mask)
        if (direction.lower() in ['f', 'fw', 'fwd', 'forward']):
            s &= (observation.visit_number == visit_number) & (observation.forward)
            self.direction = 'forward'
        elif (direction.lower() in ['b', 'bw', 'bwd', 'backward', 'back']):
            s &= (observation.visit_number == visit_number) & (~observation.forward)
            self.direction = 'backward'
        else:
            raise ValueError('No direction {}'.format(direction))

        if not (observation.exptime[s] == np.median(observation.exptime[s])).all():
           warnings.warn('Not all files have the same exposure time. {}/{} files will be discarded.'.format((observation.exptime[s] != np.median(observation.exptime[s])).sum(), len(observation.exptime[s])))

        s &= observation.exptime == np.median(observation.exptime[s])
        self.cadence_mask = s
        orbits = observation.orbits[s]
        self.orbits = orbits[:, np.any(orbits, axis=0)]

        if s.sum() == 0:
            raise ValueError('No valid cadences remaining.')
        [setattr(self, key, getattr(observation, key)[s])
            for key in ['time', 'start_date', 'propid', 'exptime', 'filters',
                        'postarg1', 'postarg2',
                        'sun_alt', 'velocity_aberration', '_filenames', 'forward', 'crpix1', 'crpix2']
                         if hasattr(observation, key)]
        setattr(self, 'hdrs', [observation.hdrs[idx] for idx in np.where(s)[0]])

        self.propid = self.propid[0]
        [setattr(self, key, getattr(observation, key))
            for key in ['ra', 'dec']
                         if hasattr(observation, key)]


        self.mask, self.spectral, self.spatial = simple_mask(observation.sci[s],
                                                             filter=self.filters[0])

        self.flat = np.ones((1, *observation.sci[s].shape[1:]))
        for count in [0, 1]:
            # We do this twice to improve the averages and flat field estimate
            self.data = (observation.sci[s]/self.flat)[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)]
            self.error = (observation.err[s]/self.flat)[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)]
            self.nt, self.nsp, self.nwav = self.data.shape
            self.shape = self.data.shape

            self.vsr_mean = average_vsr(self)
            self.spec_mean = average_spectrum(self)
            self.model = self.spec_mean * self.vsr_mean
            self.model /= np.atleast_3d(self.model.mean(axis=(1, 2))).transpose([1, 0, 2])
            self.model_err = np.zeros(self.model.shape)
            if count == 0:
                self.flat = get_flatfield(observation, self, self.model)


        T = (np.atleast_3d(self.time) * np.ones(self.shape).transpose([1, 0, 2])).transpose([1, 0, 2])
        T -= self.time[0]
        self.T = (T/T.max() - 0.5)

        Y, X = np.mgrid[:self.shape[1], :self.shape[2]]
        Y = Y/(self.shape[1] - 1) - 0.5
        X = X/(self.shape[2] - 1) - 0.5

        self.X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.data.shape)
        self.Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(self.data.shape)


        self.error[self.error/self.data > 0.1] = 1e10

        self.vsr_grad = fit_vsr_slant(self)

        m = (self.spec_mean * (self.vsr_mean + (self.vsr_grad)) * np.atleast_3d(self.average_lc).transpose([1, 0, 2]))
        med = np.median(self.data - m, axis=0)[None, :, :]
        m += med

        # Ignore a 2 pixel border when looking for cosmics
        k = (np.abs(self.Y) > -self.Y[0][2][0])
        k |= (np.abs(self.X) > -self.X[0][0][2])

        res = np.ma.masked_array(self.data - m, k)

        cmr = sigma_clip(res, axis=0, sigma_upper=6, sigma_lower=0).mask
        cmr &= ~k
        cmr |= np.asarray([(np.asarray(np.gradient(c.astype(float))) != 0).any(axis=0) for c in cmr])

        self.cosmic_rays = cmr

        self.error += self.cosmic_rays * 1e10

        self.vsr_grad = fit_vsr_slant(self)


        w = np.ones(observation.sci[s].shape)
        w[observation.err[s]/observation.sci[s] > 0.1] = 1e10

        larger_mask = convolve(self.mask[0], Box2DKernel(21)) > 1e-5

        thumb = np.average(observation.sci[s], weights=1/observation.err[s], axis=0)
        thumb = thumb[:, np.where(larger_mask.any(axis=0))[0][0]:np.where(larger_mask.any(axis=0))[0][-1] + 1]
        thumb = thumb[np.where(larger_mask.any(axis=1))[0][0]:np.where(larger_mask.any(axis=1))[0][-1] + 1, :]
        self.trace = np.mean(thumb, axis=0)
        self.trace_mask = (larger_mask.any(axis=0) & ~self.mask[0].any(axis=0))[larger_mask.any(axis=0)]

        Y, X = np.mgrid[:observation.sci[s].shape[1], :observation.sci[s].shape[2]]
        X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(observation.sci[s].shape)
        Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(observation.sci[s].shape)
        w1 = (observation.sci[s]/w)[:, larger_mask]
        w1 = (w1.T/w1.mean(axis=1)).T
        xshift = np.mean(X[:, larger_mask] * w1, axis=1)
        self.xshift = xshift - np.median(xshift)
        yshift = np.mean(Y[:, larger_mask] * w1, axis=1)
        self.yshift = yshift - np.median(yshift)


        #m1 = convolve((observation.sci[s] < 5).any(axis=0), Box2DKernel(20), boundary='fill', fill_value=1) < 0.9
        m1 = larger_mask
        m = m1 | (observation.err[s]/observation.sci[s] > 0.5)
        bkg = sigma_clipped_stats(ma.masked_array(observation.sci[s], m), axis=(1, 2), sigma=3)[1]
        self.bkg = bkg - np.median(bkg)


        self.wavelength, self.sensitivity, self.sensitivity_t = calibrate(self)
        self.wavelength = self.wavelength[~self.trace_mask]
        self.sensitivity = self.sensitivity[~self.trace_mask]
        self.sensitivity_t = self.sensitivity_t[~self.trace_mask]

    def __repr__(self):
        return '{} Visit {}, {} [Proposal ID: {}]'.format(self.name, self.visit_number, self.direction, self.propid)


    @property
    def average_lc(self):
        lc = np.average((self.data/self.model), weights=(self.model/self.error), axis=(1, 2))
        return lc

    @property
    def average_lc_errors(self):
        s = np.random.normal(self.data, self.error, size=(30, *self.shape))
        yerr = np.std([np.average(s1, weights=1/self.error, axis=(1, 2)) for s1 in s], axis=0)
        return yerr

    @property
    def raw_average_lc(self):
        return np.average((self.data/self.spec_mean), weights=(self.spec_mean/self.error), axis=(1, 2))

    @property
    def raw_channel_lcs(self):
        return np.average((self.data/self.spec_mean), weights=(self.spec_mean/self.error), axis=(1))

    @property
    def residuals(self):
        r = (self.data / self.model)
        r = (r.T - np.average(r, weights=self.model/self.error, axis=(1, 2))).T
        return r

    def animate_residuals(self, output=None):
        m = ((self.vsr_mean + (self.vsr_grad))* np.atleast_3d(self.average_lc).transpose([1, 0, 2]))
        res = (self.data / m - self.partial_model)
        k = ((np.abs(self.Y) < -self.Y[0][2][0])) * ~self.cosmic_rays

        v = np.max(np.abs(np.percentile(res, [3, 97])))

        if output is None:
            output = f'{self.name}_resids.mp4'
        animate((res/k), vmin=-v, vmax=v, cmap='coolwarm', output=output)


    def plot_channel_lcs(self):
        m = ((self.vsr_mean + (self.vsr_grad))* np.atleast_3d(self.average_lc).transpose([1, 0, 2]))
        res = (self.data / m - self.partial_model)
        k = ((np.abs(self.Y) < -self.Y[0][2][0])) * ~self.cosmic_rays

        a = (np.sum(res * k, axis=1)/np.sum(k, axis=1)).T
        vmin, vmax = np.percentile(a, [3, 97])

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot2grid((5, 1), (0, 0))

        self.plot_average_lc(ax=ax, model_lc=False)
        ax.set(title=f'{self.__repr__()}', xlabel='')

        ax = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
        dt = np.median(np.diff(self.time))
        masks = [np.in1d(np.arange(self.nt), m)
                    for m in np.array_split(np.arange(self.nt), np.where(np.append(0, np.diff(self.time)) > dt*1.5)[0])]
        for m in masks:
            plt.pcolormesh(self.time[m], self.wavelength, a[:, m], vmin=vmin, vmax=vmax, cmap='PRGn')

        bad = np.where(np.gradient(self.ts.mask.astype(float)) != 0)[0][::2]
        plt.fill_between(self.time, self.wavelength[0], self.wavelength[bad[0]], color='k', alpha=0.3)
        plt.fill_between(self.time, self.wavelength[bad[-1]], self.wavelength[-1], color='k', alpha=0.3)

        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('$\delta$ Transit Depth')
        ax.set(xlim = (self.time[0] - 0.01, self.time[-1] + 0.01), xlabel='Time [BKJD]', ylabel='Wavelength [A]')

        return fig



    def plot_average_lc(self, ax=None, errors=False, labels=True, model_lc=True):
        if ax is None:
            fig, ax1 = plt.subplots()
        else:
            ax1 = ax

        if errors:
            rlc, rlce = self.raw_average_lc, self.raw_average_lc_errors
            lc, lce = self.average_lc, self.average_lc_errors
            lce /= np.median(lc)
            lc /= np.median(lc)
            rlce /= np.median(rlc)
            rlc /= np.median(rlc)
            ax1.errorbar(self.time, rlc, rlce, ls='', c='r', label='Raw Light Curve')
            ax1.errorbar(self.time, lc, lce, ls='', c='k', label='Corrected Light Curve')

        else:
            rlc = self.raw_average_lc
            lc = self.average_lc
            lc /= np.median(lc)
            rlc /= np.median(rlc)
            ax1.scatter(self.time, rlc, c='r', marker='.', label='Raw Light Curve')
            ax1.scatter(self.time, lc, c='k', marker='.', label='Corrected Light Curve')
        ax1.set(xlabel='Time from Observation Start [d]', ylabel='Normalized Flux', title='Channel Averaged Light Curve')
        t = np.linspace(self.time[0], self.time[-1], 1000)
        if hasattr(self, 'model_lc') & model_lc:
            ax1.scatter(self.time, self.model_lc)
        if labels:
            ax1.legend()
        if ax is None:
            return fig
        else:
            return ax1

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

class Observation(object):

    def __init__(self, name):
        self.name = name
        [setattr(self, key, None)
                for key in ['time', 'start_date', 'propid', 'exptime', 'filters',
                            'hdrs', 'postarg1', 'postarg2', 'visits',
                            'sun_alt', 'velocity_aberration', '_filenames', 'sci',
                            'err', 'dq', 'forward', 'orbits',
                             'ra', 'dec', 'nt', 'ns', 'sys', 'bkg',
                              'model_lc', 'model_lc_no_ld', 'wl_map_soln', 'trace', 'crpix1', 'crpix2']]

    @staticmethod
    def from_file(filenames, propid=None, visit=None, name=None, t0=None):
        """HST Observation.

        Parameters
        ----------
        filenames : list of str, or str
            List of filenames of HST WFC3 observations, or path to a directory.
        """
        self = Observation(None)
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

        self.hdrs = [fits.getheader(file) for idx, file in enumerate(self.filenames)]
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


        convert_time = lambda hdr:(Time(datetime.strptime('{}-{}'.format(hdr['DATE-OBS'],
                                                              hdr['TIME-OBS']), '%Y-%m-%d-%H:%M:%S')).jd,
								   Time(datetime.strptime('{}'.format(hdr['DATE-OBS']), '%Y-%m-%d')).jd)

        dimage_time = []
        dimage_x = []
        dimage_y = []
        dimage_crpix1 = []
        dimage_crpix2 = []
        dimage_filter = []
        dimage_filenames = []

        for f in self.filenames[~mask]:
            with fits.open(f) as hdu:
                Y, X = np.mgrid[:hdu[1].data.shape[0], :hdu[1].data.shape[1]]
                dimage_filter.append(hdu[0].header['FILTER'])
                dimage_time.append(convert_time(hdu[0].header)[0])
                w = hdu[1].data
                cent = np.unravel_index(np.argmax(hdu[1].data), hdu[1].data.shape)
                a = np.in1d(np.arange(hdu[1].data.shape[0]), np.arange(cent[0] - 5, cent[0] + 5))
                b = np.in1d(np.arange(hdu[1].data.shape[1]), np.arange(cent[1] - 5, cent[1] + 5))
                w *= (a[:, None] * b[None, :]).astype(float)
                dimage_x.append(np.average(X, weights=w))
                dimage_y.append(np.average(Y, weights=w))
                dimage_crpix1.append(hdu[1].header['CRPIX1'])
                dimage_crpix2.append(hdu[1].header['CRPIX2'])
                dimage_filenames.append(f)

        self.dimage_time, self.dimage_x, self.dimage_y = np.asarray(dimage_time), np.asarray(dimage_x), np.asarray(dimage_y)
        self.dimage_filter = np.asarray(dimage_filter)
        self.dimage_crpix1 = np.asarray(dimage_crpix1)
        self.dimage_crpix2 = np.asarray(dimage_crpix2)
        self.dimage_filenames = np.asarray(dimage_filenames)


        self.exptime = np.asarray([hdr['EXPTIME'] for hdr in self.hdrs])
        #if not (self.exptime[mask] == np.median(self.exptime[mask])).all():
        #    warnings.warn('Not all files have the same exposure time. {}/{} files will be discarded.'.format((self.exptime[mask] != np.median(self.exptime[mask])).sum(), len(self.exptime[mask])))
        #mask &= self.exptime == np.median(self.exptime)

        self.propid = np.asarray([hdr['PROPOSID'] for hdr in self.hdrs])
        if propid is None:
            if len(np.unique(self.propid)) != 1:
                warnings.warn('Data sets from multiple proposals have been passed ({})'.format(np.unique(self.propid)))
        else:
            mask &= self.propid == propid

        if not mask.any():
            raise ValueError('All cadences are masked.')


        self.time, self.start_date = np.asarray([convert_time(hdr) for hdr in self.hdrs]).T
        s = np.argsort(self.time)
        [setattr(self, key, getattr(self, key)[s])
                for key in ['time', 'start_date', 'propid', 'exptime', 'filters', 'filenames']]

        setattr(self, 'hdrs', [self.hdrs[idx] for idx in s])

        mask = mask[s]

#        start_dates = np.unique(self.start_date)[np.append(True, np.diff(np.unique(self.start_date)) > 0.8)]
#        self.nvisits = len(start_dates)
#        self.visit_number = np.vstack([(np.abs(self.start_date - s) < 0.8) * (idx + 1)  for idx, s in enumerate(start_dates)]).sum(axis=0)

        split = np.where(np.diff(self.time) > 0.5)[0] + 1
        self.nvisits = len(split) + 1
        self.visit_number = np.hstack([np.ones(len(s), int) * (idx + 1) for idx, s in enumerate(np.array_split(self.time, split))])
        if visit is not None:
            #if visit > self.nvisits:
            #    raise ValueError('Only {} visits available.'.format(self.nvisits))
            mask &= np.in1d(self.visit_number, visit)
        self.time, self.start_date, self.visit_number, self.propid, self.exptime, self.filters =\
            self.time[mask], self.start_date[mask], self.visit_number[mask], self.propid[mask],\
            self.exptime[mask], self.filters[mask]
        self.postarg1 = np.asarray([hdr['POSTARG1'] for hdr in self.hdrs])[mask]
        self.postarg2 = np.asarray([hdr['POSTARG2'] for hdr in self.hdrs])[mask]
        self.sun_alt = np.asarray([hdr['SUN_ALT'] for hdr in self.hdrs])[mask]
        setattr(self, 'hdrs', [self.hdrs[idx] for idx in np.where(mask)[0]])
#        self.hdrs = self.hdrs[mask]

        self.ra = self.hdrs[0]['RA_TARG']
        self.dec = self.hdrs[0]['DEC_TARG']
        self.nt = np.sum(mask)
        aperture = self.hdrs[0]['APERTURE']
        self.ns = int(''.join([a for a in aperture if a.isnumeric()]))
        if self.ns == 1024:
            self.ns = 1014
        self._filenames = np.asarray(self.filenames)[mask]
        self.forward = self.postarg2 > 0

        self._load_data()
        orbits = np.where(np.append(0, np.diff(self.time)) > 0.015)[0]
        self.orbits = np.asarray([np.in1d(np.arange(self.nt), o) for o in np.array_split(np.arange(self.nt), orbits)]).T

        return self

    def build(self, cadence_mask=None):
        if cadence_mask is None:
            cadence_mask = np.ones(self.nt, bool)
        self.visits = []
        for idx in range(1, self.nvisits + 1):
            for direction, mask in zip(['f', 'b'], [self.forward, ~self.forward]):
                if mask.sum() == 0:
                    continue
                try:
                    self.visits.append(Visit(self, idx, direction, cadence_mask=cadence_mask))
                except ValueError:
                    continue

#        literature_calibrate(self)

    def _load_data(self):
        """ Helper function to load in the data """
        self.sci, self.err, self.dq = np.zeros((self.nt, self.ns, self.ns)), np.zeros((self.nt, self.ns, self.ns)), np.zeros((self.nt, self.ns, self.ns), dtype=int)
        self.velocity_aberration = np.zeros(self.nt)
        self.crpix1 = np.zeros(self.nt)
        self.crpix2 = np.zeros(self.nt)
        for jdx, file in enumerate(self._filenames):
            hdulist = fits.open(file)
            self.sci[jdx], self.err[jdx], self.dq[jdx] = np.asarray([hdu.data for hdu in hdulist[1:4]])
            if hdulist[1].header['BUNIT'] == 'ELECTRONS':
                self.sci[jdx] /= hdulist[1].header['SAMPTIME']
                self.err[jdx] /= hdulist[1].header['SAMPTIME']
            self.velocity_aberration[jdx] = hdulist[1].header['VAFACTOR']
            self.crpix1[jdx] = hdulist[1].header['CRPIX1']
            self.crpix2[jdx] = hdulist[1].header['CRPIX2']

            hdulist.close('all')
        qmask = 1 | 2 | 4 | 8 | 16 | 32 | 256
        self.err[(self.dq & qmask) != 0] = 1e10

    def __len__(self):
        return self.nt

    @property
    def shape(self):
        return self.sci.shape

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, s):
        return self.visits[s]

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
#            start_time = np.asarray([Time(datetime.strptime('{}'.format(fits.open(fname)[0].header['DATE-OBS']), '%Y-%m-%d')).jd for fname in paths])
            hdrs = np.asarray([fits.getheader(file) for idx, file in enumerate(paths)])

            convert_time = lambda hdr:(Time(datetime.strptime('{}-{}'.format(hdr['DATE-OBS'],
                                                                  hdr['TIME-OBS']), '%Y-%m-%d-%H:%M:%S')).jd,
    								   Time(datetime.strptime('{}'.format(hdr['DATE-OBS']), '%Y-%m-%d')).jd)


            time, start_date = np.asarray([convert_time(hdr) for hdr in hdrs]).T

#            start_dates = np.unique(start_date)[np.append(True, np.diff(np.unique(start_date)) > 1)]
            visits = np.unique(start_date, return_inverse=True)[1] + 1
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
        return Observation.from_file(paths, **kwargs)

    def __repr__(self):
        if self.visits is not None:
            return ('{} (WFC3 Observation) \n\t'.format(self.name)
            + '\n\t'.join([v.__repr__() for v in self.visits]))
        return '{} (WFC3 Observation)'.format(self.name)


    def fit_transit(self, sample=True, fit_period=True):
        t = [v.time for v in self.visits]
        lcs = [v.average_lc/v.average_lc.mean() for v in self.visits]
        lcs_err = [v.average_lc_errors/v.average_lc.mean() for v in self.visits]
        orbits = [v.orbits for v in self.visits]
        xs = [v.xshift for v in self.visits]
        bkg = [v.bkg for v in self.visits]

        r = fit_transit(t=t, lcs=lcs, lcs_err=lcs_err, orbits=orbits, xshift=xs, background=bkg,
                        r=self.sys.secondaries[0].r.eval(),
                        t0_val=self.sys.secondaries[0].t0.eval(),
                        period_val=self.sys.secondaries[0].porb.eval(),
                        r_star=self.sys.primary.r.eval(),
                        m_star=self.sys.primary.m.eval(),
                        exptime=np.median(self.exptime) / (3600 * 24),
                        sample=sample, fit_period=fit_period)
        if sample:
            self.wl_map_soln = r[0]
            self.trace = r[1]
        else:
            self.wl_map_soln = r

        if 'period' in self.wl_map_soln:
            self.sys.secondaries[0].porb = self.wl_map_soln['period']
        self.sys.secondaries[0].t0 = self.wl_map_soln['t0']

        self.model_lc = self.wl_map_soln['light_curve']
        self.model_lc_no_ld = self.wl_map_soln['no_limb']

        model_lc = [self.wl_map_soln['light_curve'][np.in1d(np.hstack(t), t[idx])] for idx in range(len(t))]
        for idx in range(len(t)):
            self[idx].model_lc = self.wl_map_soln['light_curve'][np.in1d(np.hstack(t), t[idx])]
            self[idx].model_lc_no_ld = self.wl_map_soln['no_limb'][np.in1d(np.hstack(t), t[idx])]

        for visit in self:
            visit.vsr_grad = fit_vsr_slant(visit)
    #    break

    def fit_transmission_spectrum(self, sample=True, npoly=3):
        ts, ts_err, model, partial_model, w, sigma_w = fit_transmission_spectrum(self, npoly=npoly)
        for idx, ts1, ts_err1 in zip(range(len(ts)), ts, ts_err):
            if ts1 is None:
                self.visits[idx].ts = None
                self.visits[idx].ts_err = None
                self.visits[idx].full_model = None
                self.visits[idx].partial_model = None
                self.visits[idx].w = None
                self.visits[idx].sigma_w = None
            else:
                k = self.visits[idx].spec_mean[0, 0] < 0.8
                self.visits[idx].ts = ma.masked_array(ts1, k)
                self.visits[idx].ts_err = ma.masked_array(ts_err1, k)
                self.visits[idx].full_model = model[idx]
                self.visits[idx].partial_model = partial_model[idx]
                self.visits[idx].w = w[idx]
                self.visits[idx].sigma_w = sigma_w[idx]
        self.munge()

    def munge(self):
        visits_numbers = [v.visit_number for v in self]
        wavelength, transmission_spec, transmission_spec_err = [], [], []
        for vn in np.unique(visits_numbers):
            vloc = np.where(visits_numbers == vn)[0]
            if len(vloc) == 2:
                w, ts, tse = [], [], []
                for vdx in vloc:
                    if self.visits[vdx].ts is None:
                        continue
                    w.append(self.visits[vdx].wavelength)
                    ts.append(self.visits[vdx].ts)
                    tse.append(self.visits[vdx].ts_err)
                if len(w) == 0:
                    continue
                elif len(w) == 1:
                    w, ts, tse = w[0], ts[0], tse[0]
                elif len(w) == 2:
                    try:
                        w, ts, tse = np.vstack(w).mean(axis=0), np.ma.mean(ts, axis=0), np.ma.hypot(*np.vstack(tse))
                    except ValueError:
                        w, ts, tse = np.hstack(w), np.ma.hstack(ts), np.ma.hstack(tse)
                        s = np.argsort(w)
                        w, ts, tse = w[s], ts[s], tse[s]

                        w = np.asarray([np.mean(a) for a in np.array_split(w, len(w)/2)])
                        ts = np.ma.masked_array([np.ma.mean(a) for a in np.array_split(ts, len(ts)/2)])
                        tse = np.ma.masked_array([(np.ma.sum(a**2)**0.5)/len(a) for a in np.array_split(tse, len(tse)/2)])

            if len(vloc) == 1:
                if self.visits[vloc[0]].ts is None:
                    continue
                w = self.visits[vloc[0]].wavelength
                ts = self.visits[vloc[0]].ts
                tse = self.visits[vloc[0]].ts_err
            wavelength.append(w)
            transmission_spec.append(ts)
            transmission_spec_err.append(tse)
        self.wavelength = wavelength
        self.transmission_spec = transmission_spec
        self.transmission_spec_err = transmission_spec_err


    @property
    def average_lc(self):
        return [v.average_lc for v in self.visits]

    @property
    def average_lc_errors(self):
        return [v.average_lc_errors for v in self.visits]

    @property
    def raw_average_lc(self):
        return [v.raw_average_lc for v in self.visits]

    @property
    def raw_channel_lcs(self):
        return [v.raw_channel_lcs for v in self.visits]


    def plot_average_lc(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for v in self.visits:
            v.plot_average_lc(ax=ax, labels=False)
        if ax is None:
            return fig
        return

    def plot_transit_fit(self):
        t0 = self.sys.secondaries[0].t0.eval()
        p = self.sys.secondaries[0].porb.eval()
        t = np.hstack([self[idx].time for idx in range(len(self.visits))])
        t_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
        mlc = (self.wl_map_soln['noise_model'] + (self.wl_map_soln['light_curve']** 0) * self.wl_map_soln['normalization'])
        fig = plt.figure(figsize=(10, 4))
        raw = np.hstack([v.average_lc/v.average_lc.mean() for v in self.visits])
        corr = np.hstack([v.average_lc/v.average_lc.mean() for v in self.visits])/mlc

        raw *= np.median(corr/raw)

        plt.scatter(t_fold, raw, c='r', s=4, label='Raw Data')
        plt.scatter(t_fold, corr, c='k', s=7, label='Corrected Data')
        plt.scatter(t_fold, self.wl_map_soln['light_curve'], c='lime', s=3, label='Transit Model')
        plt.xlabel("Time")
        plt.ylabel('Normalized Flux')
        plt.legend()
        plt.title(f'{self.name}')
        return fig
        #plt.scatter(t_fold, map_soln['light_curve'] * map_soln['normalization'], s=1, c='r')
    #
    # def plot_resids(self):
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = plt.subplot2grid((5, 1), (0, 0))
    #
    #     self.plot_average_lc(ax=ax, model_lc=False)
    #     ax.set(xlim = (self.time[0] - 0.01, self.time[-1] + 0.01), xlabel='')
    #
    #     ax = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
    #
    #
    #     ts = self.channel_lcs
    #     ts = (ts.T/np.mean(ts, axis=1))
    #
    #     vmin, vmax = np.percentile(ts, [5, 95])
    #
    #     dt = np.median(np.diff(self.time))
    #     masks = [np.in1d(np.arange(self.nt), m)
    #                 for m in np.array_split(np.arange(self.nt), np.where(np.append(0, np.diff(self.time)) > dt*1.5)[0])]
    #     for m in masks:
    #         plt.pcolormesh(self.time[m], self.wavelength, ts[:, m], vmin=vmin, vmax=vmax, cmap='PRGn')
    #     cbar = plt.colorbar(orientation='horizontal')
    #     cbar.set_label('$\delta$ Transit Depth')
    #     ax.set(xlim = (self.time[0] - 0.01, self.time[-1] + 0.01), xlabel='Time [BKJD]', ylabel='Wavelength [A]')
    #     return fig

    #
    # def plot_transmission_spectrum(self, offset=0.0005):
    #     fig = plt.figure(figsize=(10, 10))
    #
    #     a = self.oot_transit & (self.time < self.time[self.in_transit].mean())
    #     b = self.oot_transit & (self.time >= self.time[self.in_transit].mean())
    #
    #     ts = self.channel_lcs
    #     ts = (ts.T/np.mean(ts, axis=1))
    #
    #     err = ts.T[~self.in_transit].std(axis=0).data
    #     err /= (~self.in_transit).sum()**0.5
    #
    #     for idx, mask, col, label in zip([0, -1, 1], [self.in_transit, a, b], ['r', 'k', 'k'], ['In Transit', 'Before Transit', 'Out of Transit']):
    #         y = np.average(ts[:, mask], axis=1)
    #         plt.errorbar(self.wavelength, 1 - y/np.median(y) + idx * offset, err, c=col)
    #         if idx == 0:
    #             transmission_spectrum = 1 - y/np.median(y)
    #             transmission_spectrum_err = err
    #
    #     plt.xlabel('Wavelength [A]')
    #     plt.ylabel('$\delta$')
    #
    #     return fig, transmission_spectrum, transmission_spectrum_err
    #
    # def plot_channel_lcs(self, offset=0.01, lines=True, residuals=False, **kwargs):
    #     c = np.nanmedian(self.channel_lcs) *  np.ones(self.nt)
    #     cmap = kwargs.pop('cmap', plt.get_cmap('coolwarm'))
    #     fig, ax = plt.subplots(1, 2, figsize=(15, 25), sharey=True)
    #     wl = self.average_lc.data
    #     rc = self.raw_channel_lcs
    #     rc /= np.median(rc)
    #     cc = self.channel_lcs.data
    #     cc /= np.median(cc)
    #
    #     if residuals:
    #         [ax[0].scatter(self.time, rc[:, kdx] + kdx * offset - wl,
    #                     c=np.ones(self.nt) * self.wavelength[kdx], s=1,
    #                     vmin=self.wavelength[0], vmax=self.wavelength[-1],
    #                     cmap=cmap)
    #                         for kdx in range(len(self.wavelength))];
    #
    #         [ax[1].scatter(self.time, cc[:, kdx] + kdx * offset - wl,
    #                     c=np.ones(self.nt) * self.wavelength[kdx], s=1,
    #                      vmin=self.wavelength[0], vmax=self.wavelength[-1],
    #                      cmap=cmap)
    #                         for kdx in range(len(self.wavelength))];
    #         if lines:
    #             [ax[1].plot(self.time, np.ones(self.nt) * kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(cc[:, kdx]) != 0)];
    #
    #     else:
    #         [ax[0].scatter(self.time, rc[:, kdx] + kdx * offset,
    #                     c=np.ones(self.nt) * self.wavelength[kdx], s=1,
    #                     vmin=self.wavelength[0], vmax=self.wavelength[-1],
    #                     cmap=cmap)
    #                         for kdx in range(len(self.wavelength))];
    #
    #         [ax[1].scatter(self.time, cc[:, kdx] + kdx * offset,
    #                     c=np.ones(self.nt) * self.wavelength[kdx], s=1,
    #                      vmin=self.wavelength[0], vmax=self.wavelength[-1],
    #                      cmap=cmap)
    #                         for kdx in range(len(self.wavelength))];
    #         if lines:
    #             [ax[1].plot(self.time, c + kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(cc[:, kdx]) != 0)];
    #     ax[1].set(xlabel='Time', ylabel='Flux', title='Corrected', yticklabels='')
    #     ax[0].set(xlabel='Time', ylabel='Flux', title='Raw', yticklabels='')
    #     plt.subplots_adjust(wspace=0.)
    #     return fig

    def write(self, output=None):

        #        break
        channel_lcs = []
        for visit in self:
            m = ((visit.vsr_mean + (visit.vsr_grad))* np.atleast_3d(visit.average_lc).transpose([1, 0, 2]))
            res = (visit.data / m - visit.partial_model)
            k = ((np.abs(visit.Y) < -visit.Y[0][2][0])) * ~visit.cosmic_rays
            a = (np.sum(res * k, axis=1)/np.sum(k, axis=1)).T
            channel_lcs.append(a)

        r = {'name':self.name,
             'time':self.time,
             'flux':np.hstack(self.average_lc),
             'flux_err':np.hstack(self.average_lc_errors),
             'raw_flux':np.hstack(self.raw_average_lc),
             'raw_flux_err':np.hstack(self.raw_average_lc),
             'model_lc': self.model_lc,
             'model_lc_no_ld':self.model_lc_no_ld,
             'wavelength':self.wavelength,
             'transmission_spectrum':self.transmission_spec,
             'transmission_spectrum_err':self.transmission_spec_err,
             'yshift':np.hstack([v.yshift for v in self]),
             'xshift':np.hstack([v.xshift for v in self]),
             'wl_map_soln':self.wl_map_soln,
             'channel_lcs':channel_lcs}

        if output is None:
            output = f'{self.name}_output.p'
        pickle.dump(r, open(output, 'wb'))



    def save(self, dir='results'):
        if not os.path.isdir('{}/{}'.format(dir, self.name)):
            os.mkdir('{}/{}'.format(dir, self.name))

        for idx, visit in enumerate(self):
            if visit.ts is None:
                continue
            fig = visit.plot_average_lc();
            fig.savefig('{}/{}/average_lc_{}.png'.format(dir, self.name, idx + 1), bbox_inches='tight', dpi=200)
            plt.close(fig)
            visit.animate_residuals(output='results/{}/residuals_{}.mp4'.format(self.name, idx + 1))
            fig = visit.plot_channel_lcs();
            fig.savefig('{}/{}/channel_lcs_{}.png'.format(dir, self.name, idx + 1), bbox_inches='tight', dpi=200)
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.plot(visit.wavelength, visit.spec_mean[0][0]/np.median(visit.spec_mean[0][0]), label='Data', c='k')
            ax.plot(visit.wavelength, visit.sensitivity_t, label='Detector Sensitivity', c='r')
            ax.set(xlabel=('Wavelength [A]'), yticks=[])
            plt.legend()
            plt.title(visit.__repr__())
            fig.savefig('{}/{}/wavelength_calibration_{}.png'.format(dir, self.name, idx + 1), bbox_inches='tight', dpi=200)
            plt.close(fig)


        fig = self.plot_transit_fit();
        fig.savefig('{}/{}/average_lc_all.png'.format(dir, self.name), bbox_inches='tight', dpi=200)
        plt.close(fig)

        for filter in np.unique(self.filters):
            fig, ax = plt.subplots()
            for idx, w, ts, tse in zip(range(len(self.wavelength)), self.wavelength, self.transmission_spec, self.transmission_spec_err):
                if self[idx].filters[0] != filter:
                    continue
                ax.errorbar(w, ts, tse, label=f'Visit {idx + 1}', lw=0.5)
            plt.title(self.name)
            plt.legend()
            plt.xlabel('Wavelength [A]')
            plt.ylabel('$\delta$ Transit Depth [ppm]')
            fig.savefig('{}/{}/ts_all_{}.png'.format(dir, self.name, filter), bbox_inches='tight', dpi=200)
            plt.close(fig)

        self.write(output='{0}/{1}/{1}_results.p'.format(dir, self.name))

    # def correct(self):
    #     self.model = fit_data(self)
