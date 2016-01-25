"""This script prepares a pandas dataframe containing the DASCH data.

It also creates a summary csv file with details on each series.
"""
import numpy as np
import pandas as pd

from astropy.table import Table, join
from astropy.time import Time


def jd2year(jd):
    """Returns the year corresponding to a julian date."""
    return int(np.floor(Time(jd, format="jd").decimalyear))


if __name__ == "__main__":
    data = pd.read_csv("APASS_J200615.5+442725.csv")  # converted from xml.gz using topcat
    # Add useful columns
    data['isodate'] = Time(data['ExposureDate'], format="jd").iso
    data['decimalyear'] = Time(data['ExposureDate'], format="jd").decimalyear

    # Schaefer rejects yellow/red data,
    # and data points with AFLAGS > 9000.
    # AFLAGS are described here:
    # http://dasch.rc.fas.harvard.edu/database.php#AFLAGS_ext

    mask = (
            (data["seriesId"] != 12) &
            (data["seriesId"] != 13) &
            (data["AFLAGS"] <= 9000)
            ) #&
            #~((data['magcal_magdep'] == 0) & (data["limiting_mag_local"] > 12.2))
            #)
    # Schaefer also applies the cuts below, but let's deal with that later
    #        (data["magcal_local_rms"] <= 0.33) &
    #        ((data["magcal_magdep"] + 0.2) < data["limiting_mag_local"])

    output_fn = "filtered-data.csv"
    print("Writing {}".format(output_fn))
    data[mask].to_csv(output_fn)

    # Write a summary of the different photographic series to a csv file
    summary = []
    for seriesid in data["seriesId"].unique():
        series_mask = mask & (data["seriesId"] == seriesid)
        if series_mask.sum() > 0:
            data[series_mask].to_csv("series/series-{}.csv".format(seriesid))
            summary.append({
                                    'seriesId': seriesid,
                                    'year_begin': jd2year(data[series_mask]["ExposureDate"].min()),
                                    'year_end': jd2year(data[series_mask]["ExposureDate"].max()),
                                    'good_datapoints': mask.sum()
                            })
    summary_tbl = Table(summary)
    series_tbl = Table.read("dasch-plate-series.csv", format="ascii.csv")
    summary_tbl = join(summary_tbl, series_tbl,
                       keys="seriesId", join_type="left")
    summary_tbl.sort("good_datapoints")
    summary_tbl.reverse()
    summary_tbl["series", "seriesId", "good_datapoints", "year_begin", "year_end", "aperture", "telescope"].write("dasch-data-summary.csv", format="ascii.csv")
