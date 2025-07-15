## William Jones
## May 28, 2024

"""
=============
Retrieve Data
=============
Download and read data from the KMA GEO-KOMPSAT-2A (GK2A) AMI instrument

Data is downloaded from Amazon Web Services and can be returned
as a file list or read as an xarray.Dataset. If the data is not
available in a local directory, it is loaded directly into memory.

https://registry.opendata.aws/noaa-gk2a-pds/

Note that only data from February 2023 onwards is available via AWS
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import pandas as pd
import s3fs

from goes2go.data import _as_xarray, _download

# NOTE: These config dict values are retrieved from __init__ and read
# from the file ${HOME}/.config/goes2go/config.toml
from . import config

# Connect to AWS public buckets
fs = s3fs.S3FileSystem(anon=True)

# Define parameter options and aliases
# ------------------------------------

_gk2a_domain = {
    "LA": ["C", "ENH", "CONUS"],
    "FD": ["F", "FULL", "FULLDISK", "FULL DISK"],
}

_gk2a_bands = dict(
    zip(
        [
            "vi004",
            "vi005",
            "vi006",
            "vi008",
            "nr013",
            "nr016",
            "sw038",
            "wv063",
            "wv069",
            "wv073",
            "ir087",
            "ir096",
            "ir105",
            "ir112",
            "ir123",
            "ir133",
        ],
        range(1, 16 + 1),
    )
)


def _check_param_inputs(**params):
    """Check the input parameters for correct name or alias.

    Specifically, check the input for product, domain, and satellite are
    in the list of accepted values. If not, then look if it has an alias.

    As there is only one GK2A satellite and only L1B data available via AWS,
    this just returns the domain
    """
    # Kinda messy, but gets the job done.
    params.setdefault("verbose", True)
    domain = params["domain"]

    ## Determine the Domain
    if domain not in _gk2a_domain:
        if isinstance(domain, str):
            domain = domain.upper()
        for key, aliases in _gk2a_domain.items():
            if domain in aliases:
                domain = key
        assert (
            domain in _gk2a_domain
        ), f"domain must be one of {list(_gk2a_domain.keys())} or an alias {list(_gk2a_domain.values())}"


    ## Determine the Product
    return domain


def _gk2a_file_df(
    region: str,
    start: datetime,
    end: datetime,
    bands: Optional[Union[str, int, list]] = None,
    refresh: bool = True, 
    ignore_missing: bool = False, 
) -> pd.DataFrame:
    """Get list of requested GK2A AMI files as pandas.DataFrame.

    Parameters
    ----------
    region : str
    start : datetime
    end : datetime
    band : None, int, or list
        Specify the ABI channels to retrieve.
    refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.
        Default True will refresh and not use a cached list.
    ignore_missing : bool
        If True, errors when trying to search for missing time periods will be 
        ignored. Default False.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of requested files
    """
    params = locals()

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    DATES = pd.date_range(f"{start:%Y-%m-%d %H:00}", f"{end:%Y-%m-%d %H:00}", freq="1h")

    # List all files for each date
    # ----------------------------
    files = []
    for DATE in DATES:
        path = f'noaa-gk2a-pds/AMI/L1B/{region}/{DATE:%Y%m/%d/%H/}'
        if ignore_missing is True:
            try:
                files += fs.ls(path, refresh=refresh)
            except FileNotFoundError:
                print(f"Ignored missing dir: {path}")
        else:
            files += fs.ls(path, refresh=refresh)

    # Build a table of the files
    # --------------------------
    df = pd.DataFrame(files, columns=["file"])
    # df[["product_mode", "satellite", "start", "end", "creation"]] = (
    df[["satellite", "sensor", "level", "band", "product_mode", "time"]] = (
        df["file"]
        .str.rsplit("/", expand=True)
        .iloc[:, -1]
        .str.rsplit(".", expand=True)
        .loc[:, 0]
        .str.rsplit("_", expand=True, n=5)
    )

    # Filter files by band number
    # ---------------------------
    if bands is not None:
        if not hasattr(bands, "__len__") or isinstance(bands, (str, bytes, bytearray)):
            bands = [bands]
        for i_band, band in enumerate(bands):
            if band not in _gk2a_bands:
                try:
                    bands[i_band] = dict(zip(_gk2a_bands.values(), _gk2a_bands.keys()))[
                        band
                    ]
                except KeyError:
                    raise ValueError(f"Band {band} is not a valid AMI channel")
        df = df.loc[df.band.isin(bands)]

    # Filter files by requested time range
    # ------------------------------------
    # Convert filename datetime string to datetime object
    df["time"] = pd.to_datetime(df.time, format="%Y%m%d%H%M")

    # Filter by files within the requested time range
    df = df.loc[(df.time >= start) & (df.time < end)].reset_index(drop=True)

    for i in params:
        df.attrs[i] = params[i]

    return df


###############################################################################
###############################################################################


def gk2a_timerange(
    start=None,
    end=None,
    recent=None,
    *,
    domain=config["timerange"].get("gk2a_domain"),
    return_as=config["timerange"].get("return_as"),
    download=config["timerange"].get("download"),
    overwrite=config["timerange"].get("overwrite"),
    save_dir=config["timerange"].get("save_dir"),
    max_cpus=config["timerange"].get("max_cpus"),
    bands=None,
    s3_refresh=config["timerange"].get("s3_refresh"),
    ignore_missing=config["timerange"].get("ignore_missing"),
    verbose=config["timerange"].get("verbose", True),
):
    """
    Get GK2A AMI data for a time range.

    Parameters
    ----------
    start, end : datetime
        Required if recent is None.
    recent : timedelta or pandas-parsable timedelta str
        Required if start and end are None. If timedelta(hours=1), will
        get the most recent files for the past hour.
    domain : {'FD', 'LA'}
        AMI scan region indicator
    return_as : {'xarray', 'filelist'}
        Return the data as an xarray.Dataset or as a list of files
    download : bool
        - True: Download the data to disk to the location set by :guilabel:`save_dir`
        - False: Just load the data into memory.
    save_dir : pathlib.Path or str
        Path to save the data.
    overwrite : bool
        - True: Download the file even if it exists.
        - False Do not download the file if it already exists
    max_cpus : int
    bands : None, int, or list
        Specify the bands you want, default is all bands
    s3_refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.

    """
    # If `start`, or `end` is a string, parse with Pandas
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)
    # If `recent` is a string (like recent='1h'), parse with Pandas
    if isinstance(recent, str):
        recent = pd.to_timedelta(recent)

    params = locals()
    domain = _check_param_inputs(**params)
    params["domain"] = domain

    check1 = start is not None and end is not None
    check2 = recent is not None
    assert check1 or check2, "🤔 `start` and `end` *or* `recent` is required"

    if check1:
        assert hasattr(start, "second") and hasattr(
            end, "second"
        ), "`start` and `end` must be a datetime object"
    elif check2:
        assert hasattr(recent, "seconds"), "`recent` must be a timedelta object"

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day.
    if recent is not None:
        start = datetime.now(timezone.utc).replace(tzinfo=None) - recent
        end = datetime.now(timezone.utc).replace(tzinfo=None)

    df = _gk2a_file_df(
        domain, start, end, bands=bands, refresh=s3_refresh, ignore_missing=ignore_missing, 
    )

    if download:
        _download(df, save_dir=save_dir, overwrite=overwrite, verbose=verbose)

    if return_as == "filelist":
        df.attrs["filePath"] = save_dir
        return df
    elif return_as == "xarray":
        return _as_xarray(df, **params)


def gk2a_latest(
    *,
    domain=config["latest"].get("domain"),
    return_as=config["latest"].get("return_as"),
    download=config["latest"].get("download"),
    overwrite=config["latest"].get("overwrite"),
    save_dir=config["latest"].get("save_dir"),
    bands=None,
    s3_refresh=config["latest"].get("s3_refresh"),
    verbose=config["latest"].get("verbose", True),
):
    """
    Get the latest available GOES data.

    Parameters
    ----------
    domain : {'FD', 'LA'}
        AMI scan region indicator
    return_as : {'xarray', 'filelist'}
        Return the data as an xarray.Dataset or as a list of files
    download : bool
        - True: Download the data to disk to the location set by :guilabel:`save_dir`
        - False: Just load the data into memory.
    save_dir : pathlib.Path or str
        Path to save the data.
    overwrite : bool
        - True: Download the file even if it exists.
        - False Do not download the file if it already exists
    bands : None, int, or list
        ONLY FOR L1b-Rad products; specify the bands you want
    s3_refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.
    """
    params = locals()
    domain = _check_param_inputs(**params)
    params["domain"] = domain

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day. Look in the current hour and last hour.
    start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    end = datetime.now(timezone.utc).replace(tzinfo=None)

    df = _gk2a_file_df(domain, start, end, bands=bands, refresh=s3_refresh)

    # Filter for specific mesoscale domain
    if domain is not None and domain.upper() in ["M1", "M2"]:
        df = df[df["file"].str.contains(f"{domain.upper()}-M")]

    # Get the most recent file (latest start date)
    df = df.loc[df.time == df.time.max()].reset_index(drop=True)

    if download:
        _download(df, save_dir=save_dir, overwrite=overwrite, verbose=verbose)

    if return_as == "filelist":
        df.attrs["filePath"] = save_dir
        return df
    elif return_as == "xarray":
        return _as_xarray(df, **params)


def gk2a_nearesttime(
    attime,
    within=pd.to_timedelta(config["nearesttime"].get("within", "1h")),
    *,
    domain=config["nearesttime"].get("domain"),
    return_as=config["nearesttime"].get("return_as"),
    download=config["nearesttime"].get("download"),
    overwrite=config["nearesttime"].get("overwrite"),
    save_dir=config["nearesttime"].get("save_dir"),
    bands=None,
    s3_refresh=config["nearesttime"].get("s3_refresh"),
    verbose=config["nearesttime"].get("verbose", True),
):
    """
    Get the GOES data nearest a specified time.

    Parameters
    ----------
    attime : datetime
        Time to find the nearest observation for.
        May also use a pandas-interpretable datetime string.
    within : timedelta or pandas-parsable timedelta str
        Timerange tht the nearest observation must be.
    domain : {'FD', 'LA'}
        AMI scan region indicator
    return_as : {'xarray', 'filelist'}
        Return the data as an xarray.Dataset or as a list of files
    download : bool
        - True: Download the data to disk to the location set by :guilabel:`save_dir`
        - False: Just load the data into memory.
    save_dir : pathlib.Path or str
        Path to save the data.
    overwrite : bool
        - True: Download the file even if it exists.
        - False: Do not download the file if it already exists
    bands : None, int, or list
        ONLY FOR L1b-Rad products; specify the bands you want
    s3_refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.
    """
    if isinstance(attime, str):
        attime = pd.to_datetime(attime)
    if isinstance(within, str):
        within = pd.to_timedelta(within)

    params = locals()
    domain = _check_param_inputs(**params)
    params["domain"] = domain

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day.
    start = attime - within
    end = attime + within

    df = _gk2a_file_df(domain, start, end, bands=bands, refresh=s3_refresh)

    # return df, start, end, attime

    # Filter for specific mesoscale domain
    # Get row that matches the nearest time
    df = df.sort_values("time")
    df = df.set_index(df.time)
    unique_times_index = df.index.unique()
    nearest_time_index = unique_times_index.get_indexer([attime], method="nearest")
    nearest_time = unique_times_index[nearest_time_index]
    df = df.loc[nearest_time]
    df = df.reset_index(drop=True)

    n = len(df.file)
    if n == 0:
        print("🛸 No data....🌌")
        return None

    if download:
        _download(df, save_dir=save_dir, overwrite=overwrite, verbose=verbose)

    if return_as == "filelist":
        df.attrs["filePath"] = save_dir
        return df
    elif return_as == "xarray":
        return _as_xarray(df, **params)
