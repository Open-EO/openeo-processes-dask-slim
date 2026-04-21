from typing import Optional

import dask.array as da
import xarray as xr
from openeo.udf import UdfData
from openeo.udf.run_code import run_udf_code
from openeo.udf.xarraydatacube import XarrayDataCube
from openeo_processes_dask_slim.process_implementations.data_model import RasterCube

__all__ = ["run_udf"]


def run_udf(
    data: da.Array, udf: str, runtime: str, context: Optional[dict] = None
) -> RasterCube:
    input_attrs = data.attrs if isinstance(data, xr.DataArray) else {}
    udf_input = XarrayDataCube(xr.DataArray(data))
    udf_data = UdfData(datacube_list=[udf_input], user_context=context)
    result = run_udf_code(code=udf, data=udf_data)
    cubes = result.get_datacube_list()
    if len(cubes) != 1:
        raise ValueError(
            f"The provided UDF should return one datacube, but got: {result}"
        )
    result_array: xr.DataArray = cubes[0].array
    if not result_array.attrs:
        result_array.attrs = input_attrs
    return result_array
