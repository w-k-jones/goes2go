## Brian Blaylock
## September 16, 2020

"""
=============
Retrieve Data.
=============

Download and read data from the R-series Geostationary Operational
Environmental Satellite data.

Data is downloaded from Amazon Web Services and can be returned
as a file list or read as an xarray.Dataset. If the data is not
available in a local directory, it is loaded directly into memory.

https://registry.opendata.aws/noaa-goes/
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from goes2go.tools import lat_lon_to_scan_angles

# NOTE: These config dict values are retrieved from __init__ and read
# from the file ${HOME}/.config/goes2go/config.toml
from . import config

# Connect to AWS public buckets
fs = s3fs.S3FileSystem(anon=True)

# Define parameter options and aliases
# ------------------------------------
_himawari_satellite = {
    "noaa-himawari8": [8, "8", "H8", "HIMAWARI8", "HIMAWARI-8"], 
    "noaa-himawari9": [9, "9", "H9", "HIMAWARI9", "HIMAWARI-9"], 
}

_himawari_domain = {
    "Japan": ["C", "CONUS", "JAPAN"],
    "FLDK": ["F", "FULL", "FULLDISK", "FULL DISK"],
    "Target": ["M", "MESOSCALE", "M1", "M2", "TARGET"],
}

_himawari_resolution = {
    "R05": [0.5, 500, "0.5", "500"], 
    "R10": [1, 1000, "1", "1000"], 
    "R20": [2, 2000, "2", "2000"], 
}

_himawari_bands = dict(
    zip(
        [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B09",
            "B10",
            "B11",
            "B12",
            "B13",
            "B14",
            "B15",
            "B16",
        ],
        range(1, 16 + 1),
    )
)

def _check_param_inputs(**params):
    """Check the input parameters for correct name or alias.

    Specifically, check the input for product, domain, and satellite are
    in the list of accepted values. If not, then look if it has an alias.
    """
    # Kinda messy, but gets the job done.
    params.setdefault("verbose", True)
    satellite = params["satellite"]
    domain = params["domain"]
    resolution = params["resolution"]
    # verbose = params["verbose"]

    ## Determine the Satellite
    if satellite not in _himawari_satellite:
        if isinstance(satellite, str):
            satellite = str(satellite).upper()
        for key, aliases in _himawari_satellite.items():
            if satellite in aliases:
                satellite = key
        if satellite not in _himawari_satellite:
            raise ValueError(f"satellite must be one of {list(_himawari_satellite.keys())} or an alias {list(_himawari_satellite.values())}")

    ## Determine the Domain
    if domain not in _himawari_domain:
        if isinstance(domain, str):
            domain = domain.upper()
        for key, aliases in _himawari_domain.items():
            if domain in aliases:
                domain = key
        if domain not in _himawari_domain:
            raise ValueError(f"domain must be one of {list(_himawari_domain.keys())} or an alias {list(_himawari_domain.values())}")

    ## Determine resolution
    if resolution not in _himawari_domain or resolution is not None:
        if isinstance(resolution, str):
            resolution = str(resolution).upper()
        for key, aliases in _himawari_resolution.items():
            if resolution in aliases:
                resolution = key
        if resolution not in _himawari_resolution:
            raise ValueError(f"resolution must be one of {list(_himawari_resolution.keys())} or an alias {list(_himawari_resolution.values())}")

    return satellite, domain, resolution


def _himawari_file_df(satellite, domain, start, end, bands=None, resolution=None, refresh=True, ignore_missing=False):
    """Get list of requested GOES files as pandas.DataFrame.

    Parameters
    ----------
    satellite : str
    product : str
    start : datetime
    end : datetime
    band : None, int, or list
        Specify the ABI channels to retrieve.
    refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.
        Default True will refresh and not use a cached list.
    """
    params = locals()

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    DATES = pd.date_range(f"{start:%Y-%m-%d %H:00}", f"{end:%Y-%m-%d %H:00}", freq="1h")

    # List all files for each date
    # ----------------------------
    files = []
    for DATE in DATES:
        path = f"{satellite}/AHI-L1b-{domain}/{DATE:%Y/%j/%H/}"
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
    df.drop(index=df.index[~df["file"].str.contains(".nc")],inplace=True)
    df[["product_mode", "satellite", "start", "end", "creation"]] = (
        df["file"].str.rsplit("_", expand=True, n=5).loc[:, 1:]
    )

    # Filter files by band number
    # ---------------------------
    if bands is not None:
        if not hasattr(bands, "__len__") or isinstance(bands, (str, bytes, bytearray)):
            bands = [bands]
        for i_band, band in enumerate(bands):
            if band not in _himawari_bands:
                try:
                    bands[i_band] = dict(zip(_himawari_bands.values(), _himawari_bands.keys()))[
                        band
                    ]
                except KeyError:
                    raise ValueError(f"Band {band} is not a valid AHI channel")
        df = df.loc[df.band.isin(bands)]

    # Filter files by resolution
    # --------------------------
    if resolutions is not None:
        if not hasattr(resolutions, "__len__") or isinstance(resolutions, (str, bytes, bytearray)):
            resolutions = [resolutions]
        for i_resolution, resolution in enumerate(resolutions):
            if resolution not in _himawari_resolution:
                for key, aliases in _himawari_resolution.items():
                    if resolution in aliases:
                        resolutions[i_resolution] = key
                        break
                else:
                    raise ValueError(f"Resolution {resolution} is not a valid AHI resolution")
        df = df.loc[df.resolution.isin(resolutions)]
    else:
        # If None pick the highest resolution for each
        df = df.loc[df.resolution == df.groupby("band").resolution.unique().str[0][df.band].to_numpy()]
    
    # Filter files by requested time range
    # ------------------------------------
    # Convert filename datetime string to datetime object
    df["start"] = pd.to_datetime(df.start, format="s%Y%j%H%M%S%f")
    df["end"] = pd.to_datetime(df.end, format="e%Y%j%H%M%S%f")
    df["creation"] = pd.to_datetime(df.creation, format="c%Y%j%H%M%S%f.nc")

    # Filter by files within the requested time range
    df = df.loc[df.start >= start].loc[df.end <= end].reset_index(drop=True)

    for i in params:
        df.attrs[i] = params[i]

    return df


def _download(df, save_dir, overwrite, max_threads=10, verbose=False):
    """Download the files from a DataFrame listing with multithreading."""

    def do_download(src):
        dst = Path(save_dir) / src
        if not dst.parent.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and not overwrite:
            if verbose:
                print(f" 👮🏻‍♂️ File already exists. Do not overwrite: {dst}")
        else:
            # Downloading file from AWS
            fs.get(src, str(dst))

    ################
    # Multithreading
    tasks = len(df)
    threads = min(tasks, max_threads)

    with ThreadPoolExecutor(threads) as exe:
        futures = [exe.submit(do_download, src) for src in df.file]

        # nothing is returned in the list
        this_list = [future.result() for future in as_completed(futures)]

    print(
        f"📦 Finished downloading [{len(df)}] files to [{save_dir/Path(df.file[0]).parents[3]}]."
    )


###############################################################################
###############################################################################


def himawari_timerange(
    start=None,
    end=None,
    recent=None,
    *,
    satellite=config["timerange"].get("satellite"),
    product=config["timerange"].get("product"),
    domain=config["timerange"].get("domain"),
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
    Get GOES data for a time range.

    Parameters
    ----------
    start, end : datetime
        Required if recent is None.
    recent : timedelta or pandas-parsable timedelta str
        Required if start and end are None. If timedelta(hours=1), will
        get the most recent files for the past hour.
    satellite : {'goes16', 'goes17', 'goes18'}
        Specify which GOES satellite.
        The following alias may also be used:

        - ``'goes16'``: 16, 'G16', or 'EAST'
        - ``'goes17'``: 17, 'G17', or 'WEST'
        - ``'goes18'``: 18, 'G18', or 'WEST'

    product : {'ABI', 'GLM', other GOES product}
        Specify the product name.

        - 'ABI' is an alias for ABI-L2-MCMIP Multichannel Cloud and Moisture Imagery
        - 'GLM' is an alias for GLM-L2-LCFA Geostationary Lightning Mapper

        Others may include ``'ABI-L1b-Rad'``, ``'ABI-L2-DMW'``, etc.
        For more available products, look at this `README
        <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_
    domain : {'C', 'F', 'M'}
        ABI scan region indicator. Only required for ABI products if the
        given product does not end with C, F, or M.

        - C: Contiguous United States (alias 'CONUS')
        - F: Full Disk (alias 'FULL')
        - M: Mesoscale (alias 'MESOSCALE')

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
        ONLY FOR L1b-Rad products; specify the bands you want
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
    satellite, product, domain = _check_param_inputs(**params)
    params["satellite"] = satellite
    params["product"] = product
    params["domain"] = domain

    check1 = start is not None and end is not None
    check2 = recent is not None
    if not (check1 or check2):
        raise ValueError("🤔 `start` and `end` *or* `recent` is required")
    if check1:
        if not (hasattr(start, "second") and hasattr(end, "second")):
            raise ValueError( "`start` and `end` must be a datetime object")
    elif check2:
        if not hasattr(recent, "seconds"):
            raise ValueError("`recent` must be a timedelta object")

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day.
    if recent is not None:
        start = datetime.utcnow() - recent
        end = datetime.utcnow()

    df = _himawari_file_df(satellite, product, start, end, bands=bands, refresh=s3_refresh, ignore_missing=ignore_missing)

    if download:
        _download(df, save_dir=save_dir, overwrite=overwrite, verbose=verbose)

    if return_as == "filelist":
        df.attrs["filePath"] = save_dir
        return df
    elif return_as == "xarray":
        raise ValueError("xarray return not yet enabled for AHI")
        # return _as_xarray(df, **params)

def _preprocess_single_point(ds, target_lat, target_lon, decimal_coordinates=True):
    """
    Preprocessing function to select only the single relevant data subset.

    Parameters
    ----------
    ds: xarray Dataset
        The dataset to look through and choose the particular location
    target_lat, target_lon : float
        Location where you wish to extract the point values from
    decimal_coordinates: bool
        If latitude/longitude are specified in decimal or radian coordinates.
    """
    x_target, y_target = lat_lon_to_scan_angles(target_lat, target_lon, ds["goes_imager_projection"], decimal_coordinates)
    return ds.sel(x=x_target, y=y_target, method="nearest")

def himawari_single_point_timerange(
    latitude,
    longitude,
    start=None,
    end=None,
    recent=None,
    decimal_coordinates=True,
    *,
    satellite=config["timerange"].get("satellite"),
    product=config["timerange"].get("product"),
    domain=config["timerange"].get("domain"),
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
    Get GOES data for a time range at the scan point nearest to a defined single latitude/longitude point.

    Parameters
    ----------
    latitude, longitude : float
            Location where you wish to extract the point values from
    start, end : datetime
        Required if recent is None.
    recent : timedelta or pandas-parsable timedelta str
        Required if start and end are None. If timedelta(hours=1), will
        get the most recent files for the past hour.
    decimal_coordinates: bool
        If latitude/longitude are specified in decimal or radian coordinates.
    satellite : {'goes16', 'goes17', 'goes18'}
        Specify which GOES satellite.
        The following alias may also be used:

        - ``'goes16'``: 16, 'G16', or 'EAST'
        - ``'goes17'``: 17, 'G17', or 'WEST'
        - ``'goes18'``: 18, 'G18', or 'WEST'

    product : {'ABI', 'GLM', other GOES product}
        Specify the product name.

        - 'ABI' is an alias for ABI-L2-MCMIP Multichannel Cloud and Moisture Imagery
        - 'GLM' is an alias for GLM-L2-LCFA Geostationary Lightning Mapper

        Others may include ``'ABI-L1b-Rad'``, ``'ABI-L2-DMW'``, etc.
        For more available products, look at this `README
        <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_
    domain : {'C', 'F', 'M'}
        ABI scan region indicator. Only required for ABI products if the
        given product does not end with C, F, or M.

        - C: Contiguous United States (alias 'CONUS')
        - F: Full Disk (alias 'FULL')
        - M: Mesoscale (alias 'MESOSCALE')

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
        ONLY FOR L1b-Rad products; specify the bands you want
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
    satellite, product, domain = _check_param_inputs(**params)
    params["satellite"] = satellite
    params["product"] = product
    params["domain"] = domain

    check1 = start is not None and end is not None
    check2 = recent is not None
    if not (check1 or check2):
        raise ValueError("🤔 `start` and `end` *or* `recent` is required")
    if check1:
        if not (hasattr(start, "second") and hasattr(end, "second")):
            raise ValueError( "`start` and `end` must be a datetime object")
    elif check2:
        if not hasattr(recent, "seconds"):
            raise ValueError("`recent` must be a timedelta object")

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day.
    if recent is not None:
        start = datetime.utcnow() - recent
        end = datetime.utcnow()

    df = _himawari_file_df(satellite, product, start, end, bands=bands, refresh=s3_refresh, ignore_missing=ignore_missing)

    if download:
        _download(df, save_dir=save_dir, overwrite=overwrite, verbose=verbose)

    if return_as == "filelist":
        df.attrs["filePath"] = save_dir
        return df
    elif return_as == "xarray":
        raise ValueError("xarray return not yet enabled for AHI")
        # partial_func = partial(_preprocess_single_point, target_lat=latitude, target_lon=longitude, decimal_coordinates=decimal_coordinates)
        # preprocessed_ds = xr.open_mfdataset([str(config['timerange']['save_dir']) + "/" + f for f in df['file'].to_list()],
        #           concat_dim='t',
        #           combine='nested',
        #           preprocess=partial_func)
        # return preprocessed_ds


def himawari_latest(
    *,
    satellite=config["latest"].get("satellite"),
    product=config["latest"].get("product"),
    domain=config["latest"].get("domain"),
    return_as=config["latest"].get("return_as"),
    download=config["latest"].get("download"),
    overwrite=config["latest"].get("overwrite"),
    save_dir=config["latest"].get("save_dir"),
    bands=None,
    s3_refresh=config["latest"].get("s3_refresh"),
    ignore_missing=config["latest"].get("ignore_missing"),
    verbose=config["latest"].get("verbose", True),
):
    """
    Get the latest available GOES data.

    Parameters
    ----------
    satellite : {'goes16', 'goes17', 'goes18'}
        Specify which GOES satellite.
        The following alias may also be used:

        - ``'goes16'``: 16, 'G16', or 'EAST'
        - ``'goes17'``: 17, 'G17', or 'WEST'
        - ``'goes18'``: 18, 'G18', or 'WEST'

    product : {'ABI', 'GLM', other GOES product}
        Specify the product name.

        - 'ABI' is an alias for ABI-L2-MCMIP Multichannel Cloud and Moisture Imagery
        - 'GLM' is an alias for GLM-L2-LCFA Geostationary Lightning Mapper

        Others may include ``'ABI-L1b-Rad'``, ``'ABI-L2-DMW'``, etc.
        For more available products, look at this `README
        <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_
    domain : {'C', 'F', 'M'}
        ABI scan region indicator. Only required for ABI products if the
        given product does not end with C, F, or M.

        - C: Contiguous United States (alias 'CONUS')
        - F: Full Disk (alias 'FULL')
        - M: Mesoscale (alias 'MESOSCALE')

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
    satellite, product, domain = _check_param_inputs(**params)
    params["satellite"] = satellite
    params["product"] = product
    params["domain"] = domain

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day. Look in the current hour and last hour.
    start = datetime.utcnow() - timedelta(hours=1)
    end = datetime.utcnow()

    df = _himawari_file_df(satellite, product, start, end, bands=bands, refresh=s3_refresh, ignore_missing=ignore_missing)

    # Filter for specific mesoscale domain
    if domain is not None and domain.upper() in ["M1", "M2"]:
        df = df[df["file"].str.contains(f"{domain.upper()}-M")]

    # Get the most recent file (latest start date)
    df = df.loc[df.start == df.start.max()].reset_index(drop=True)

    if download:
        _download(df, save_dir=save_dir, overwrite=overwrite, verbose=verbose)

    if return_as == "filelist":
        df.attrs["filePath"] = save_dir
        return df
    elif return_as == "xarray":
        raise ValueError("xarray return not yet enabled for AHI")
        # return _as_xarray(df, **params)


def himawari_nearesttime(
    attime,
    within=pd.to_timedelta(config["nearesttime"].get("within", "1h")),
    *,
    satellite=config["nearesttime"].get("satellite"),
    product=config["nearesttime"].get("product"),
    domain=config["nearesttime"].get("domain"),
    return_as=config["nearesttime"].get("return_as"),
    download=config["nearesttime"].get("download"),
    overwrite=config["nearesttime"].get("overwrite"),
    save_dir=config["nearesttime"].get("save_dir"),
    bands=None,
    s3_refresh=config["nearesttime"].get("s3_refresh"),
    ignore_missing=config["nearesttime"].get("ignore_missing"),
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
    satellite : {'goes16', 'goes17', 'goes18'}
        Specify which GOES satellite.
        The following alias may also be used:

        - ``'goes16'``: 16, 'G16', or 'EAST'
        - ``'goes17'``: 17, 'G17', or 'WEST'
        - ``'goes18'``: 18, 'G18', or 'WEST'

    product : {'ABI', 'GLM', other GOES product}
        Specify the product name.

        - 'ABI' is an alias for ABI-L2-MCMIP Multichannel Cloud and Moisture Imagery
        - 'GLM' is an alias for GLM-L2-LCFA Geostationary Lightning Mapper

        Others may include ``'ABI-L1b-Rad'``, ``'ABI-L2-DMW'``, etc.
        For more available products, look at this `README
        <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_
    domain : {'C', 'F', 'M'}
        ABI scan region indicator. Only required for ABI products if the
        given product does not end with C, F, or M.

        - C: Contiguous United States (alias 'CONUS')
        - F: Full Disk (alias 'FULL')
        - M: Mesoscale (alias 'MESOSCALE')

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
    satellite, product, _ = _check_param_inputs(**params)
    params["satellite"] = satellite
    params["product"] = product

    # Parameter Setup
    # ---------------
    # Create a range of directories to check. The GOES S3 bucket is
    # organized by hour of day.
    start = attime - within
    end = attime + within

    df = _himawari_file_df(satellite, product, start, end, bands=bands, refresh=s3_refresh, ignore_missing=ignore_missing)

    # return df, start, end, attime

    # Filter for specific mesoscale domain
    if domain.upper() in ["M1", "M2"]:
        df = df[df["file"].str.contains(f"{domain.upper()}-M")]

    # Get row that matches the nearest time
    df = df.sort_values("start")
    df = df.set_index(df.start)
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
        raise ValueError("xarray return not yet enabled for AHI")
        # return _as_xarray(df, **params)
