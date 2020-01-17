import argparse
import pickle

from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path, PosixPath

import geopandas as gpd
import numpy as np
import salem
import xarray

from affine import Affine
from rasterio import features
from tqdm import tqdm

shp_file = '/home/frederik/Downloads/basin_shps_simplyfied/basin_shps_simplyfied.shp'
nc_files = [
    '/home/frederik/Downloads/macav2livneh_pr_CCSM4_r6i1p1_rcp85_2086_2099_CONUS_monthly.nc'
    '/home/frederik/Downloads/macav2livneh_tasmax_CCSM4_r6i1p1_rcp85_2086_2099_CONUS_daily.nc'
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_file',
                        type=str,
                        required=True,
                        help="Full path to the shape file containing the basin boundaries")
    parser.add_argument('--nc_folder',
                        type=str,
                        required=True,
                        help="Path to the folder, containing the netCDF files.")
    parser.add_argument('--num_threads',
                        type=int,
                        required=True,
                        help="Number of worker processing netCDF files in parallel")

    return vars(parser.parse_args())


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes: list,
              coords: xarray.core.coordinates.DatasetCoordinates,
              latitude='lat',
              longitude='lon',
              fill=np.nan,
              **kwargs):
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(
        shapes,
        out_shape=out_shape,
        fill=fill,
        transform=transform,
        dtype=float,
        all_touched=True,
        **kwargs,
    )
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xarray.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def get_averages(nc_file: PosixPath, shp_file: PosixPath, out_dir: PosixPath):
    shdf = salem.read_shapefile(shp_file)
    xr = xarray.open_dataset(nc_file)
    # shift lon so it matches the shape file's data range
    xr['lon'] -= 360

    # empty dict to store the data
    data = {}

    for n, df in shdf.iterrows():

        # subset xarray to basin boundaries for faster processing
        arr = xr.sel(lon=slice(df.min_x - 1, df.max_x + 1), lat=slice(df.min_y - 1, df.max_y + 1))

        # rasterize basin polygon
        arr['basin'] = rasterize([df.geometry], arr.coords, longitude='lon', latitude='lat')

        # select xarray data only within the rasterized polygon
        arr_sub = arr.where(~np.isnan(arr.basin), other=np.nan)

        data[str(df.hru_id).zfill(8)] = arr_sub.mean(dim=['lat', 'lon'])

    parts = nc_file.stem.split('_')

    climate_model_dir = out_dir / parts[0]
    climate_model_dir.mkdir(parents=True, exist_ok=True)

    out_file = climate_model_dir / "_".join(parts[1:]) + '.p'
    with out_file.open("rb") as fp:
        pickle.dump(data)


def create_jobs(cfg: dict) -> list:
    pass


def process_files(cfg: dict):
    pass


if __name__ == "__main__":
    args = get_args()
    process_files(cfg=args)
