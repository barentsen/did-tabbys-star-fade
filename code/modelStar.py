import os
import glob

from scipy.stats import norm
import scipy.optimize as op
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import emcee
import corner

from astropy import time
from astropy import log


class OffsetModel(object, ):

    def __init__(self, df, noDetectionCutoffPercentile = 10, yerrLimit=0.1, yerrMultiplier=1.0):
        """ Expecting a pandas data frame """
        self.noDetectionCutoffPercentile = noDetectionCutoffPercentile
        # Add the decimalyear column if necessary
        if "decimalyear" not in df:
            df['decimalyear'] = time.Time(df['ExposureDate'], format="jd").decimalyear

        self.isNonDetection = df['magcal_magdep'] == 0   # No detection
        # Data points should be considered bad if:
        # - they are flagged bath (AFLAGS)
        # - they were obtained using yellow or red filters (series 12 and 13);
        # - they provide an upper limit fainter than the median magnitude.
        self.magCut = np.percentile(df['magcal_magdep'][~self.isNonDetection], 
            self.noDetectionCutoffPercentile)
        log.info("mag cut: {}, and {}th percentile".format(self.magCut, 
            self.noDetectionCutoffPercentile))
        self.maskBad = (
                    (df["AFLAGS"] > 9000) |
                    (df["seriesId"] == 12) |
                    (df["seriesId"] == 13) |
                    (self.isNonDetection & (df['limiting_mag_local'] > self.magCut))
                )
        log.info("There are {} detections and {} non-detections;"
                 "{} are bad.".format(self.isNonDetection.sum(),
                                      (~self.isNonDetection).sum(),
                                      self.maskBad.sum()))

        self.x = df['decimalyear'][~self.maskBad & ~self.isNonDetection].values
        self.y = df['magcal_magdep'][~self.maskBad & ~self.isNonDetection].values
        self.yerr = df['magcal_local_rms'][~self.maskBad & ~self.isNonDetection].values
        self.yerr *= yerrMultiplier

        self.xLimit = df['decimalyear'][(~self.maskBad) & self.isNonDetection].values
        self.yLimit = df['limiting_mag_local'][(~self.maskBad) & self.isNonDetection].values
        self.yerrLimit = yerrLimit
        log.info("yerr limit = {}".format(yerrLimit))

        self.series = df['seriesId'][(~self.maskBad) & self.isNonDetection].values
        self.nSeries = np.shape(np.unique(self.series))[0]
        self.uniqueSeries = np.unique(self.series)

    @staticmethod
    def _lnlike(theta, obsdata):
        x, y, yerr = obsdata
        m, b, lnf = theta
        model = m * x + b

        npt_lc = np.shape(y)[0]
        err_jit2 = yerr**2 + (np.e**lnf)**2

        loglc = (
            - (npt_lc/2.)*np.log(2.*np.pi)
            - 0.5 * np.sum(np.log(err_jit2))
            - 0.5 * np.sum((model - y)**2 / err_jit2)
            )

        return loglc

    @staticmethod
    def _lnlikeLimit(theta, obsdata):
        m, b, lnfu = theta
        model = m * x + b
        err_jit2 = yerr**2 + (np.e**lnfu)**2

        npt_lc = np.shape(y)[0]
        loglc = (-npt_lc * np.log(2) + 
            np.sum(np.log(1 + scipy.special.erf((model - y) / 
                (err_jit2**0.5 * 2**0.5)))))

        return loglc


    def lnprior(self, theta):
        m = theta[0]
        b = theta[1:1+self.nSeries]
        lnf = theta[-2]
        lnfu = theta[-1]

        if (-2 <= m <= 2.) and np.all(np.abs(b) <= 100. ) and \
            (-10 <= lnf <= 10.) and (-10 <= lnfu <= 10.):
            return 0.0

        return -np.inf


    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        lnlike = 0.0
        m = theta[0]
        b = theta[1:1+self.nSeries]
        lnf = theta[-2]
        lnfu = theta[-1]
        for i, series in enumerate(self.uniqueSeries):
            mask = self.series == series
            maskObs = mask
            lnlike += self._lnlike([m, b[i], lnf], 
                [self.x[mask], self.y[mask], self.yerr[mask]])
            lnlike += self._lnlikeLimit([m, b[i], lnfu], 
                [self.xLimit[mask], self.yLimit[mask], self.yerrLimit])

        return lp + lnlike


    def mcmc(self, nwalkers=100, nsamps=3000, burnin=500, threads=1,
        initialM=0.01, initialB=12., initialF=0.01, initialFu=0.01):
        """Samples the slope and intercept using MCMC."""
        log.info("Calling EnsembleSampler.run_mcmc")

        ndim = self.nSeries + 3
        pos = [np.r_[initialM, np.repeat(initialB, self.nSeries), initialF, initialFu]
            + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                    threads=threads)

        _ = self.sampler.run_mcmc(pos, nsamps)
        samples = self.sampler.chain[:, burnin:, :].reshape((-1, ndim))
        return samples
        


if __name__ == '__main__':
    om = OffsetModel()

    samples = om.mcmc()










