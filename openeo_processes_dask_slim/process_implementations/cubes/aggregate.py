import copy
import gc
import logging
from typing import Callable, Optional, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xvec
from joblib import Parallel, delayed
from openeo_pg_parser_networkx.pg_schema import TemporalInterval, TemporalIntervals

from openeo_processes_dask_slim.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    TooManyDimensions,
)

__all__ = ["aggregate_temporal", "aggregate_temporal_period"]

logger = logging.getLogger(__name__)


def aggregate_temporal(
    data: RasterCube,
    intervals: Union[TemporalIntervals, list[TemporalInterval], list[Optional[str]]],
    reducer: Callable,
    labels: Optional[list] = None,
    dimension: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        t = dimension
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        t = temporal_dims[0]
    if isinstance(intervals, TemporalIntervals) or isinstance(intervals, list):
        interval_str = []
        for interval in intervals:
            if isinstance(interval, TemporalInterval):
                interval_0 = str(interval[0].root)
                interval_1 = str(interval[1].root)
                interval_str.append([interval_0, interval_1])
        if interval_str:
            intervals = interval_str

    intervals_np = (
        np.array(intervals, dtype=np.datetime64).astype("datetime64[s]").astype(float)
    )
    intervals_flat = np.reshape(
        intervals_np, np.shape(intervals_np)[0] * np.shape(intervals_np)[1]
    )

    if not labels:
        labels = np.array(intervals, dtype="datetime64[s]").astype(str)[:, 0]
    if (intervals_np[1:, 0] < intervals_np[:-1, 1]).any():
        raise NotImplementedError(
            "Aggregating data for overlapping time ranges is not implemented. "
        )

    mask = np.zeros((len(labels) * 2) - 2).astype(bool)
    mask[1::2] = np.isin(intervals_np[1:, 0], intervals_np[:-1, 1])
    mask = np.append(mask, np.array([False, True]))

    labels_nans = np.arange(len(labels) * 2).astype(str)
    labels_nans[::2] = labels
    labels_nans = labels_nans[~mask]

    intervals_flat = np.unique(intervals_flat)
    data_copy = copy.deepcopy(data)
    t_coords = data_copy[t].values.astype(str)
    data_copy[t] = np.array(t_coords, dtype="datetime64[s]").astype(float)
    grouped_data = data_copy.groupby_bins(t, bins=intervals_flat)
    positional_parameters = {"data": 0}
    groups = grouped_data.reduce(
        reducer, keep_attrs=True, positional_parameters=positional_parameters
    )
    groups[t + "_bins"] = labels_nans
    data_agg_temp = groups.sel({t + "_bins": labels})
    data_agg_temp = data_agg_temp.rename({t + "_bins": t})

    return data_agg_temp


def get_intervals(data, period):
    start, end = data["t"].values[0], data["t"].values[-1]

    year_start = int(start.astype("datetime64[Y]").astype(int)) + 1970
    year_end = int(end.astype("datetime64[Y]").astype(int)) + 1970
    month_start = int(start.astype("datetime64[M]").astype(int)) % 12 + 1
    month_end = int(end.astype("datetime64[M]").astype(int)) % 12 + 1
    day_start_val = (
        int(
            (
                start.astype("datetime64[D]")
                - start.astype("datetime64[M]").astype("datetime64[D]")
            ).astype(int)
        )
        + 1
    )

    if period == "decade":
        dec_start = int(np.floor(year_start / 10) * 10)
        dec_end = int(np.ceil(year_end / 10) * 10)
        decade_years = list(range(dec_start, dec_end + 1, 10))
        intervals = [f"{y:04d}-01-01T00:00:00" for y in decade_years]
        labels = [f"{y:04d}" for y in decade_years[:-1]]

    elif period == "decade-ad":
        dec_start = int(np.floor(year_start / 10) * 10) + 1
        dec_end = int(np.ceil(year_end / 10) * 10) + 1
        decade_years = list(range(dec_start, dec_end + 1, 10))
        intervals = [f"{y:04d}-01-01T00:00:00" for y in decade_years]
        labels = [f"{y:04d}" for y in decade_years[:-1]]

    elif period == "tropical-season":
        if month_start >= 5 and month_start < 10:
            ts_year, ts_month = year_start, 5
        elif month_start < 5:
            ts_year, ts_month = year_start - 1, 11
        else:
            ts_year, ts_month = year_start, 11

        if month_end >= 5 and month_end < 10:
            te_year, te_month = year_end, 11
        elif month_end < 5:
            te_year, te_month = year_end, 5
        else:
            te_year, te_month = year_end + 1, 5

        intervals = []
        y, m = ts_year, ts_month
        while (y < te_year) or (y == te_year and m <= te_month):
            intervals.append(f"{y:04d}-{m:02d}-01T00:00:00")
            m += 6
            if m > 12:
                m -= 12
                y += 1

        labels = []
        for interval in intervals[:-1]:
            if "-11-" in interval:
                labels.append(interval[:5] + "ndjfma")
            if "-05-" in interval:
                labels.append(interval[:5] + "mjjaso")

    elif period == "dekad":
        dekad_day_start = int(np.floor((day_start_val - 1) / 10) * 10 + 1)
        end_dt = end.astype("datetime64[D]")

        intervals = []
        cur_year, cur_month, cur_day = year_start, month_start, dekad_day_start
        while True:
            intervals.append(f"{cur_year:04d}-{cur_month:02d}-{cur_day:02d}T00:00:00")
            cur_dt = np.datetime64(f"{cur_year:04d}-{cur_month:02d}-{cur_day:02d}", "D")
            if cur_dt > end_dt:
                break
            if cur_day == 1:
                cur_day = 11
            elif cur_day == 11:
                cur_day = 21
            else:
                cur_month += 1
                if cur_month > 12:
                    cur_month = 1
                    cur_year += 1
                cur_day = 1

        labels = []
        for interval in intervals[:-1]:
            dt = np.datetime64(interval[:10], "D")
            year_val = int(dt.astype("datetime64[Y]").astype(int)) + 1970
            day_of_year = (
                int((dt - np.datetime64(f"{year_val:04d}-01-01", "D")).astype(int)) + 1
            )
            dekad = int(day_of_year / 10)
            labels.append(f"{year_val}-{dekad:02d}")

    else:
        raise NotImplementedError(
            f"The provided period '{period})' is not implemented. "
        )

    interval_array = np.array(intervals, dtype=str)
    interval_matrix = np.zeros((len(interval_array) - 1, 2)).astype(str)
    interval_matrix[:, 0] = interval_array[:-1]
    interval_matrix[:, 1] = interval_array[1:]
    return interval_matrix, list(labels)


def aggregate_temporal_period(
    data: RasterCube,
    reducer: Callable,
    period: str,
    dimension: Optional[str] = None,
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        applicable_temporal_dimension = temporal_dims[0]

    periods_to_frequency = {
        "hour": "h",
        "day": "D",
        "week": "W",
        "month": "ME",
        "season": "QS-DEC",
        "year": "YS",
    }

    if period in periods_to_frequency.keys():
        times = data[applicable_temporal_dimension].values

        if period == "hour":
            group_keys = times.astype("datetime64[h]").astype("datetime64[us]")
            t_min = times.min().astype("datetime64[h]")
            t_max = times.max().astype("datetime64[h]")
            all_periods = np.arange(
                t_min, t_max + np.timedelta64(1, "h"), np.timedelta64(1, "h")
            ).astype("datetime64[us]")
        elif period == "day":
            group_keys = times.astype("datetime64[D]").astype("datetime64[us]")
            t_min = times.min().astype("datetime64[D]")
            t_max = times.max().astype("datetime64[D]")
            all_periods = np.arange(
                t_min, t_max + np.timedelta64(1, "D"), np.timedelta64(1, "D")
            ).astype("datetime64[us]")
        elif period == "week":
            days = times.astype("datetime64[D]").astype(int)
            weekday = (days + 3) % 7
            monday_int = days - weekday
            group_keys = monday_int.astype("datetime64[D]").astype("datetime64[us]")
            all_periods = (
                np.arange(monday_int.min(), monday_int.max() + 7, 7)
                .astype("datetime64[D]")
                .astype("datetime64[us]")
            )
        elif period == "month":
            months_int = times.astype("datetime64[M]").astype(int)
            group_keys = months_int.astype("datetime64[M]").astype("datetime64[us]")
            all_periods = (
                np.arange(months_int.min(), months_int.max() + 1)
                .astype("datetime64[M]")
                .astype("datetime64[us]")
            )
        elif period == "season":
            months_int = times.astype("datetime64[M]").astype(int)
            month_in_year = months_int % 12
            year = months_int // 12 + 1970
            season_month = np.where(
                month_in_year == 11,
                11,
                np.where(
                    month_in_year <= 1,
                    11,
                    np.where(month_in_year <= 4, 2, np.where(month_in_year <= 7, 5, 8)),
                ),
            )
            season_year = np.where(month_in_year <= 1, year - 1, year)
            group_keys = np.array(
                [
                    np.datetime64(f"{y:04d}-{m+1:02d}", "M").astype("datetime64[us]")
                    for y, m in zip(season_year, season_month)
                ]
            )
            min_m = group_keys.min().astype("datetime64[M]").astype(int)
            max_m = group_keys.max().astype("datetime64[M]").astype(int)
            season_start_months = {2, 5, 8, 11}
            all_periods = np.array(
                [
                    np.datetime64(
                        f"{(m // 12 + 1970):04d}-{(m % 12 + 1):02d}", "M"
                    ).astype("datetime64[us]")
                    for m in range(min_m, max_m + 1)
                    if m % 12 in season_start_months
                ]
            )
        elif period == "year":
            years_int = times.astype("datetime64[Y]").astype(int)
            group_keys = years_int.astype("datetime64[Y]").astype("datetime64[us]")
            all_periods = (
                np.arange(years_int.min(), years_int.max() + 1)
                .astype("datetime64[Y]")
                .astype("datetime64[us]")
            )

        label_da = xr.DataArray(
            group_keys,
            dims=[applicable_temporal_dimension],
            name=applicable_temporal_dimension,
        )
        label_da = label_da.drop_vars(applicable_temporal_dimension, errors="ignore")

        positional_parameters = {"data": 0}
        result = data.groupby(label_da).reduce(
            reducer, keep_attrs=True, positional_parameters=positional_parameters
        )
        return result.reindex({applicable_temporal_dimension: all_periods})

    else:
        intervals, labels = get_intervals(data, period)
        return aggregate_temporal(
            data=data, intervals=intervals, reducer=reducer, labels=labels
        )
