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

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.convolution import convolve, Box2DKernel

from astroquery.mast import Observations as astropyObs

from .methods import simple_mask, average_vsr, average_spectrum, get_flatfield, calibrate, build_vsr, build_spectrum, fit_data, fit_transit

from starry.extensions import from_nexsci

log = logging.getLogger('ombre')

class Visit(object):

    def __init__(self, observation, visit_number, direction='forward'):
        self.visit_number = visit_number
        self.name = observation.name

        if (direction.lower() in ['f', 'fw', 'fwd', 'forward']):
            s = (observation.visit_number == visit_number) & (observation.forward)
            self.direction = 'forward'
        elif (direction.lower() in ['b', 'bw', 'bwd', 'backward', 'back']):
            s = (observation.visit_number == visit_number) & (~observation.forward)
            self.direction = 'backward'
        else:
            raise ValueError('No direction {}'.format(direction))

        self.cadence_mask = s
        orbits = observation.orbits[s]
        self.orbits = orbits[:, np.any(orbits, axis=0)]
        [setattr(self, key, getattr(observation, key)[s])
            for key in ['time', 'start_date', 'propid', 'exptime', 'filters',
                        'hdrs', 'postarg1', 'postarg2',
                        'sun_alt', 'velocity_aberration', '_filenames', 'forward']
                         if hasattr(observation, key)]
        self.propid = self.propid[0]
        [setattr(self, key, getattr(observation, key))
            for key in ['ra', 'dec', 'teff']
                         if hasattr(observation, key)]
        self.mask, self.spectral, self.spatial = simple_mask(observation.sci[s])

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

        self.error[self.error/self.data > 0.1] = 1e10
        res = (self.data / self.model)
        res = ((res.T) - np.mean(res, axis=(1, 2))).T
        self.cosmic_rays = sigma_clip(res, sigma=8, axis=0).mask
        self.error += self.cosmic_rays * 1e10


        w = np.ones(observation.sci[s].shape)
        w[observation.err[s]/observation.sci[s] > 0.1] = 1e10


        larger_mask = convolve(self.mask[0], Box2DKernel(11)) > 1e-5
        Y, X = np.mgrid[:observation.sci[s].shape[1], :observation.sci[s].shape[2]]
        X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(observation.sci[s].shape)
        Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(observation.sci[s].shape)
        w1 = (observation.sci[s]/w)[:, larger_mask]
        w1 = (w1.T/w1.mean(axis=1)).T
        xshift = np.mean(X[:, larger_mask] * w1, axis=1)
        self.xshift = xshift - np.median(xshift)
        yshift = np.mean(Y[:, larger_mask] * w1, axis=1)
        self.yshift = yshift - np.median(yshift)

        m1 = convolve((observation.sci[s] < 5).any(axis=0), Box2DKernel(20), boundary='fill', fill_value=1) < 0.9
        m = m1 | (observation.err[s]/observation.sci[s] > 0.5)
        bkg = sigma_clipped_stats(ma.masked_array(observation.sci[s], m), axis=(1, 2), sigma=3)[1]
        self.bkg = bkg - np.median(bkg)

        T = (np.atleast_3d(self.time) * np.ones(self.shape).transpose([1, 0, 2])).transpose([1, 0, 2])
        T -= self.time[0]
        self.T = (T/T.max() - 0.5)

        Y, X = np.mgrid[:self.shape[1], :self.shape[2]]
        Y = Y/(self.shape[1] - 1) - 0.5
        X = X/(self.shape[2] - 1) - 0.5

        self.X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.data.shape)
        self.Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(self.data.shape)

        self.sensitivity, self.wavelength = calibrate(self, self.teff)

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

    def plot_average_lc(self, ax=None, errors=False, labels=True):
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
#        if self.wl_map_soln is not None:
#            ax1.scatter(self.time, self.wl_map_soln['light_curve'] + 1)
#        ax1.plot(t, self.sys.flux(t).eval(), c='b', label='model')
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
                             'ra', 'dec', 'nt', 'ns', 'sys', 'teff', 'bkg',
                              'model_lc', 'model_lc_no_ld', 'wl_map_soln', 'trace']]

    @staticmethod
    def from_file(filenames, propid=None, visit=None, teff=6000, name=None, t0=None, load_only=False):
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

        start_dates = np.unique(self.start_date)[np.append(True, np.diff(np.unique(self.start_date)) > 1)]
        self.nvisits = len(start_dates)
        self.visit_number = np.vstack([(np.abs(self.start_date - s) < 2) * (idx + 1)  for idx, s in enumerate(start_dates)]).sum(axis=0)

        if visit != None:
            if visit > self.nvisits:
                raise ValueError('Only {} visits available.'.format(self.nvisits))
            mask &= self.visit_number == visit
        self.time, self.start_date, self.visit_number, self.propid, self.exptime, self.filters =\
            self.time[mask], self.start_date[mask], self.visit_number[mask], self.propid[mask],\
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
        if self.ns == 1024:
            self.ns = 1014
        self._filenames = np.asarray(self.filenames)[mask]
        self.forward = self.postarg2 > 0

        self._load_data()
        orbits = np.where(np.append(0, np.diff(self.time)) > 0.015)[0]
        self.orbits = np.asarray([np.in1d(np.arange(self.nt), o) for o in np.array_split(np.arange(self.nt), orbits)]).T

        if load_only:
            return self

        self.visits = []
        for idx in range(1, self.nvisits + 1):
            for direction, mask in zip(['f', 'b'], [self.forward, ~self.forward]):
                if mask.sum() == 0:
                    continue
                self.visits.append(Visit(self, idx, direction))
        return self

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
            start_dates = np.unique(self.start_date)[np.append(True, np.diff(np.unique(self.start_date)) > 1)]
            visits = np.asarray([np.where(np.sort(np.unique(start_date)) == t1)[0][0] + 1 for t1 in start_dates])
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


    def fit_transit(self, sample=True):
        if sample:
            self.wl_map_soln, self.trace = fit_transit(self, sample=sample)
        else:
            self.wl_map_soln = fit_transit(self, sample=sample)
        self.model_lc = self.wl_map_soln['light_curve'] + 1
        self.model_lc_no_ld = self.wl_map_soln['no_limb'] + 1

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


    def plot_average_lc(self):
        fig, ax = plt.subplots()
        for v in self.visits:
            v.plot_average_lc(ax=ax, labels=False)
        return fig

    def plot_resids(self):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot2grid((5, 1), (0, 0))

        self.plot_average_lc(ax=ax)
        ax.set(xlim = (self.time[0] - 0.01, self.time[-1] + 0.01), xlabel='')

        ax = plt.subplot2grid((5, 1), (1, 0), rowspan=4)


        ts = self.channel_lcs
        ts = (ts.T/np.mean(ts, axis=1))

        vmin, vmax = np.percentile(ts, [5, 95])

        dt = np.median(np.diff(self.time))
        masks = [np.in1d(np.arange(self.nt), m)
                    for m in np.array_split(np.arange(self.nt), np.where(np.append(0, np.diff(self.time)) > dt*1.5)[0])]
        for m in masks:
            plt.pcolormesh(self.time[m], self.wavelength, ts[:, m], vmin=vmin, vmax=vmax, cmap='PRGn')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('$\delta$ Transit Depth')
        ax.set(xlim = (self.time[0] - 0.01, self.time[-1] + 0.01), xlabel='Time [BKJD]', ylabel='Wavelength [A]')
        return fig


    def plot_transmission_spectrum(self, offset=0.0005):
        fig = plt.figure(figsize=(10, 10))

        a = self.oot_transit & (self.time < self.time[self.in_transit].mean())
        b = self.oot_transit & (self.time >= self.time[self.in_transit].mean())

        ts = self.channel_lcs
        ts = (ts.T/np.mean(ts, axis=1))

        err = ts.T[~self.in_transit].std(axis=0).data
        err /= (~self.in_transit).sum()**0.5

        for idx, mask, col, label in zip([0, -1, 1], [self.in_transit, a, b], ['r', 'k', 'k'], ['In Transit', 'Before Transit', 'Out of Transit']):
            y = np.average(ts[:, mask], axis=1)
            plt.errorbar(self.wavelength, 1 - y/np.median(y) + idx * offset, err, c=col)
            if idx == 0:
                transmission_spectrum = 1 - y/np.median(y)
                transmission_spectrum_err = err

        plt.xlabel('Wavelength [A]')
        plt.ylabel('$\delta$')

        return fig, transmission_spectrum, transmission_spectrum_err

    def plot_channel_lcs(self, offset=0.01, lines=True, residuals=False, **kwargs):
        c = np.nanmedian(self.channel_lcs) *  np.ones(self.nt)
        cmap = kwargs.pop('cmap', plt.get_cmap('coolwarm'))
        fig, ax = plt.subplots(1, 2, figsize=(15, 25), sharey=True)
        wl = self.average_lc.data
        rc = self.raw_channel_lcs
        rc /= np.median(rc)
        cc = self.channel_lcs.data
        cc /= np.median(cc)

        if residuals:
            [ax[0].scatter(self.time, rc[:, kdx] + kdx * offset - wl,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                        vmin=self.wavelength[0], vmax=self.wavelength[-1],
                        cmap=cmap)
                            for kdx in range(len(self.wavelength))];

            [ax[1].scatter(self.time, cc[:, kdx] + kdx * offset - wl,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                         vmin=self.wavelength[0], vmax=self.wavelength[-1],
                         cmap=cmap)
                            for kdx in range(len(self.wavelength))];
            if lines:
                [ax[1].plot(self.time, np.ones(self.nt) * kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(cc[:, kdx]) != 0)];

        else:
            [ax[0].scatter(self.time, rc[:, kdx] + kdx * offset,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                        vmin=self.wavelength[0], vmax=self.wavelength[-1],
                        cmap=cmap)
                            for kdx in range(len(self.wavelength))];

            [ax[1].scatter(self.time, cc[:, kdx] + kdx * offset,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                         vmin=self.wavelength[0], vmax=self.wavelength[-1],
                         cmap=cmap)
                            for kdx in range(len(self.wavelength))];
            if lines:
                [ax[1].plot(self.time, c + kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(cc[:, kdx]) != 0)];
        ax[1].set(xlabel='Time', ylabel='Flux', title='Corrected', yticklabels='')
        ax[0].set(xlabel='Time', ylabel='Flux', title='Raw', yticklabels='')
        plt.subplots_adjust(wspace=0.)
        return fig


    def correct(self):
        self.model = fit_data(self)
