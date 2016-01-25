import os
import glob

from scipy.stats import norm
import scipy.optimize as op
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import emcee

from astropy import time
from astropy import log


def lnlike(theta,  x,  y, yerr):
    """Returns the log likelihood."""
    m, b = theta
    model = m * x + b
    return np.sum(norm.logpdf(model - y, scale=yerr))

def lnlike_limit(theta, x_limit, y_limit, yerr_limit=0.1):
    """Non-detections."""
    m, b = theta
    model = m * x_limit + b
    return np.sum(norm.logcdf(model - y_limit, scale=yerr_limit))

def lnprior(theta):
    """Returns the log prior."""
    m, b = theta
    if -1. < m < 1. and -100.0 < b < 100.0:
        return 0.0  # log(Probability=1) = 0
    return -np.inf  # log(Probability=verysmall) = -inf


def lnprob(theta, x, y, yerr, x_limit, y_limit, yerr_limit):
    """Returns the log posterior."""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp +
            lnlike(theta, x, y, yerr) +
            lnlike_limit(theta, x_limit, y_limit, yerr_limit))


def mcmc(x, y, yerr, x_limit, y_limit, yerr_limit,
         nwalkers=30, nsamp=3000, burnin=100, corner_plot=True):
    """Samples the slope and intercept using MCMC."""
    log.info("Calling EnsembleSampler.run_mcmc")
    # Initialize the model at a good starting position
    ndim = 2
    pos = [np.array([0.01, 12.]) + 1e-4*np.random.randn(ndim)
           for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x, y, yerr, x_limit, y_limit, yerr_limit), threads=10)
    _ = sampler.run_mcmc(pos, 1000)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples


def run_analysis(df, name="default", ymin=13., ymax=11.8):
    """"""
    # Add the decimalyear column if necessary
    if "decimalyear" not in df:
        df['decimalyear'] = time.Time(df['ExposureDate'], format="jd").decimalyear

    mask_nodetection = df['magcal_magdep'] == 0   # No detection
    # Data points should be considered bad if:
    # - they are flagged bath (AFLAGS)
    # - they were obtained using yellow or red filters (series 12 and 13);
    # - they provide an upper limit fainter than the median magnitude.
    mag_10p = np.percentile(df['magcal_magdep'][~mask_nodetection], 10)
    log.info("10p mag: {}".format(mag_10p))
    mask_bad = (
                (df["AFLAGS"] > 9000) |
                (df["seriesId"] == 12) |
                (df["seriesId"] == 13) |
                (mask_nodetection & (df['limiting_mag_local'] > mag_10p))
            )
    log.info("There are {} detections and {} non-detections;"
             "{} are bad.".format(mask_nodetection.sum(),
                                  (~mask_nodetection).sum(),
                                  mask_bad.sum()))

    x = df['decimalyear'][~mask_bad & ~mask_nodetection].values
    y = df['magcal_magdep'][~mask_bad & ~mask_nodetection].values
    yerr = df['magcal_local_rms'][~mask_bad & ~mask_nodetection].values

    x_limit = df['decimalyear'][(~mask_bad) & mask_nodetection].values
    y_limit = df['limiting_mag_local'][(~mask_bad) & mask_nodetection].values
    yerr_limit = np.median(yerr)
    log.info("yerr_limit = {}".format(yerr_limit))
    samples = mcmc(x, y, yerr, x_limit, y_limit, yerr_limit)
    samples_m, samples_b = samples[:, 0], samples[:, 1]

    lightcurve_plot(x, y, yerr, samples_m, samples_b,
                    ymin=ymin, ymax=ymax,
                    output_fn="output/{}.png".format(name))
    corner_plot(samples, output_fn="output/{}-corner.png".format(name))

    # Summarize the posterior using different statistics
    m_mean, m_std = np.mean(samples_m), np.std(samples_m)
    b_mean, b_std = np.mean(samples_b), np.std(samples_b)
    # Probability that the slope is positive
    prob_slope_positive = (samples_m > 0).sum() / samples_m.size
    # Probability that the slope is negative
    prob_slope_negative = (samples_m < 0).sum() / samples_m.size
    # Probability that the slope is less than 0.05 mag per century
    prob_slope_0p05 = (samples_m < 0.0005).sum() / samples_m.size
    # Write the results to a text file
    out = open("output/mcmc-result-{}.txt".format(name), "w")
    out.write("Results for {}\n=====================\n".format(name))
    out.write("Data points: {}\n".format(len(x)))
    out.write("Period: {:.1f} - {:.1f}\n".format(x.min(), x.max()))
    out.write("MCMC samples: {}\n\n".format(len(samples_m)))

    msg_mcmc = ("Mean and standard deviation of the posterior parameters:\n"
                "m = {:.5f} +/- {:.5f}\n"
                "b = {:.3f} +/- {:.3f}\n\n"
                .format(m_mean, m_std, b_mean, b_std))
    out.write(msg_mcmc)
    msg_verbose = ("i.e. the star changed by:\n"
                   "{0:+.3f} +/_ {1:.3f} mag/century\n"
                   "= {0:+.2f} +/_ {1:.2f} mag/century\n\n".format(100*m_mean, 100*m_std))
    out.write(msg_verbose)
    out.write("P(slope > 0): {:.1f}%\n".format(100 * prob_slope_positive))
    out.write("P(slope < 0): {:.1f}%\n".format(100 * prob_slope_negative))
    out.write("P(slope < 0.0005): {:.1f}%\n".format(100 * prob_slope_0p05))
    out.close()
    # And write some info to the terminal to update the user
    log.info(msg_mcmc)
    log.info(msg_verbose)
    del samples


def corner_plot(samples, output_fn="corner_plot.png"):
    import corner
    fig = corner.corner(samples, labels=["$m$", "$b$"], )
    log.info("Writing {}".format(output_fn))
    fig.savefig(output_fn)
    pl.close(fig)


def lightcurve_plot(x, y, yerr, samples_m, samples_b, ymin=13., ymax=11.8, output_fn="plot.png"):
    fig = pl.figure(figsize=(7, 2.5))
    ax = fig.add_subplot(1, 1, 1)

    #ax.scatter(x, y, marker="o", s=1, lw=0, facecolor="black")
    #ax.errorbar(x, y, yerr=yerr, fmt=".k", ms=1, capsize=0, alpha=0.3, elinewidth=0.5)
    ax.plot(x, y, ".k", ms=2)

    n_draws = 100  # Number of draws to visualize the posterior in data space
    xval = np.arange(x.min(), x.max(), 1)
    savearr = np.zeros([n_draws, len(xval)])
    for idx, samples_idx in enumerate(np.random.randint(len(samples_m), size=n_draws)):
        savearr[idx] = samples_m[samples_idx] * xval + samples_b[samples_idx]
    pc = np.percentile(savearr, [16, 50, 84], axis=0)
    fill = True
    if fill:
        ax.fill_between(xval, pc[0], pc[2], alpha=0.5)
    else:
        for m, b in samples[np.random.randint(len(samples), size=100)]:
            ax.plot(xl, m * xl + b, color="k", alpha=0.05)
    ax.minorticks_on()
    ax.set_xlim(1885, 1995)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(1890, 1991, 10))
    ax.set_xlabel('$\mathrm{Year}$')
    ax.set_ylabel('B')
    fig.tight_layout(pad=0.4)
    fig.savefig(output_fn, dpi=200)
    pl.close(fig)


if __name__ == "__main__":
    """
    for csv_fn in glob.glob("../data/nearby-blue-stars/*.csv"):
        log.info("Fitting " + csv_fn)
        df = pd.read_csv(csv_fn)
        run_analysis(df, name=os.path.basename(csv_fn))
        del df
    """

    df = pd.read_csv("../data/filtered-data.csv")
    run_analysis(df, name="all")

    """
    mask = (df['seriesId'] != 11) & (df['seriesId'] != 12) & (df['seriesId'] != 13)
    run_analysis(df[mask], name="without-series-11-12-13")
    """

    """
    for seriesId in df['seriesId'].unique():
        mask = df['seriesId'] == seriesId
        #if mask.sum() >= 20:  # Ignore series with less than 10 points
        run_analysis(df[mask], name="series-{}".format(seriesId))
    """
    """
    for seriesId in df['seriesId'].unique():
        mask = df['seriesId'] != seriesId
        if mask.sum() >= 5:
            run_analysis(df[mask], name="without-series-{}".format(seriesId))
    """


    #df = pd.read_csv("../data/hd190717/preprocessed-data.csv")
    #run_analysis(df, name="hd190717", ymin=11.5, ymax=10.)
    
    #mask = (df['seriesId'] != 11)
    #run_analysis(df[mask], name="hd190717-without-series-11")