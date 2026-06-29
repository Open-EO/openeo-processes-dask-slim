import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

logger = logging.getLogger(__name__)


def create_fake_rastercube(
    data,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    backend="numpy",
    chunks=("auto", "auto", "auto", -1),
):
    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    x_coords = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        step=len_x / data.shape[0],
    )
    y_coords = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        step=len_y / data.shape[1],
    )

    t_start = np.datetime64(str(temporal_extent.root[0].root), "us")
    t_end = np.datetime64(str(temporal_extent.root[1].root), "us")
    t_coords = (
        np.linspace(t_start.astype(np.int64), t_end.astype(np.int64), data.shape[2])
        .astype(np.int64)
        .astype("datetime64[us]")
    )

    coords = {"x": x_coords, "y": y_coords, "t": t_coords, "bands": bands}

    raster_cube = xr.DataArray(
        data=data,
        coords=coords,
        attrs={"crs": spatial_extent.crs},
    )
    import odc.geo.xr

    raster_cube = odc.geo.xr.assign_crs(raster_cube, crs=spatial_extent.crs)

    if "dask" in backend:
        import dask.array as da

        raster_cube.data = da.from_array(raster_cube.data, chunks=chunks)

    return raster_cube
