import os

os.environ["KMP_WARNINGS"] = "off"

import argparse
import geopandas as gpd
import hashlib
import numpy as np
import pandas as pd
import pickle
import rasterio as rio

from numba import jit, prange
from pathlib import Path
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import mapping
from tqdm import tqdm


def df_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float64 columns to float32.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame to convert.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame with float64 columns converted to float32.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


@jit(nopython=True)
def binary_search(arr, x):
    """
    Perform binary search of sorted array arr for x.
    Return the index of x in arr if present, else -1.
    """
    low = 0
    high = arr.size - 1

    while low <= high:
        mid = (low + high) // 2
        mid_val = arr[mid]

        if mid_val < x:
            low = mid + 1
        elif mid_val > x:
            high = mid - 1
        else:
            return mid  # x is found
    return -1  # x is not found


@jit(nopython=True, parallel=True)
def jit_inun(hand, seg_catch, hydroids, stage_m):
    inun = np.empty_like(hand, dtype=np.float32)
    inun.fill(np.nan)
    for i in prange(hand.shape[0]):
        for j in prange(hand.shape[1]):
            hydroid = seg_catch[i, j]
            hand_h = hand[i, j]
            hydroid_idx = binary_search(hydroids, hydroid)
            if hydroid_idx != -1:
                h = stage_m[hydroid_idx]
            else:
                h = -9999
            if h > hand_h:
                inun[i, j] = h - hand_h
    return inun


def get_file_metadata_hash(filepath):
    # Get file size and modification timestamp
    file_stats = os.stat(filepath)
    metadata_string = f"{file_stats.st_size}_{file_stats.st_mtime}"
    # Generate a hash based on the file metadata
    return hashlib.sha256(metadata_string.encode()).hexdigest()[:8]


def clip_raster_by_gdf(raster_path, gdf, id_column=None, return_max=False):
    """
    read raster_path with rasterio
    clip to the geometry of gdf
    store clipped rasters and transforms
    as tuples in a dictionary with id_column as key (row index if None)
    """
    # Open the raster file
    with rio.open(raster_path) as ds:
        h_max_all = np.nanmax(ds.read(1))
        out_image_dict = {}
        for idx, row in gdf.iterrows():
            # The geometry must be in GeoJSON format
            geom = [mapping(row["geometry"])]
            # Perform the clipping
            out_image, out_transform = mask(ds, geom, crop=True)
            out_image[out_image == ds.nodata] = np.nan
            # Store both the numpy array and its transform in the dictionary
            r_key = idx if id_column is None else row[id_column]
            out_image_dict[r_key] = (out_image[0], out_transform)

    return (out_image_dict, h_max_all) if return_max else out_image_dict


def get_flood_stage_without_table(
    geometry_vol,
    volume_geometry_path,
    hand_path,
    cell_area=1,
    constrained=False,
):
    """
    Calculate the flood stage for each polygon in gdf.

    Parameters
    ----------
    geometry_vol : `geopandas.GeoDataFrame`
        GeoDataFrame containing the polygons to calculate flood stage for.
        Must have a column with volume of ponded water in each polygon.
    volume_geometry_path : `str`
        Path to the vector geometry within which to spread ATS ponded water.
    hand_path : `str`
        Path to the HAND raster.
    cell_area : `float`
        Area of each cell in the HAND raster. Calculated as xres*yres:
        abs(profile["transform"].a * profile["transform"].e)
    constrained : `bool`, optional
        Whether to use the constrained flood stage calculation. Default is False.
        If True, the flood stage is constrained by the maximum elevation in the HAND raster.

    Returns
    -------
    flood_stage : `geopandas.GeoDataFrame`
        GeoDataFrame with the calculated flood stage and inundation volume.
    """

    # h_max_all = np.nanmax(hand)
    hand_dict, h_max_all = clip_raster_by_gdf(
        hand_path, geometry_vol, return_max=True
    )

    for idx, row in geometry_vol.iterrows():
        # print(idx)
        hand_clipped = hand_dict[idx][0]
        vol = row["ponded_wat_vol"]
        if (vol == 0) | np.all(np.isnan(hand_clipped)):
            geometry_vol.loc[idx, "handvol"] = 0
            geometry_vol.loc[idx, "H"] = 0
        else:
            h = 0.25
            h_target = np.nanmax(hand_clipped) if constrained else h_max_all
            while h < h_target:
                # array of inundation depth
                inun_h = h - hand_clipped
                inun_h[inun_h < 0] = 0
                # convert to volume by summing all cells and multiplying by cell area
                inun_vol = np.nansum(inun_h) * cell_area
                if inun_vol > vol:
                    break
                h += 0.25
            geometry_vol.loc[idx, "handvol"] = inun_vol
            geometry_vol.loc[idx, "H"] = h

    # constrained_str = "constrained" if constrained else "unconstrained"

    # gdf.index.name = "quad_idx"
    return df_float64_to_float32(geometry_vol.drop(columns=["geometry"]))


def get_flood_stage(
    geometry_vol,
    volume_geometry_path,
    hand_path,
    cell_area,
    constrained=False,
    custom_stage_vol_path=None,
):
    """
    Calculate the flood stage for each polygon in gdf.

    Parameters
    ----------
    geometry_vol : `geopandas.GeoDataFrame`
        GeoDataFrame containing the polygons to calculate flood stage for.
        Must have a "ponded_wat_vol" column.
    volume_geometry_path : `str`
        Path to the vector geometry within which to spread ATS ponded water.
    hand_path : `str`
        Path to the HAND raster.
    cell_area : `float`
        Area of each cell in the HAND raster. Calculated as xres*yres:
        abs(profile["transform"].a * profile["transform"].e)
    constrained : `bool`, optional
        Whether to use the constrained flood stage calculation. Default is False.
        If True, the flood stage is constrained by the maximum elevation in the HAND raster.

    Returns
    -------
    flood_stage : `geopandas.GeoDataFrame`
        GeoDataFrame with the calculated flood stage and inundation volume.
    """

    if custom_stage_vol_path:
        stage_vol_path = custom_stage_vol_path
    else:
        # get unique 8 char hash based on file metadata
        # unique to both files' size and modification timestamp
        hash1 = get_file_metadata_hash(volume_geometry_path)
        hash2 = get_file_metadata_hash(hand_path)
        unique_hash = hashlib.sha256((hash1 + hash2).encode()).hexdigest()[:8]
        stage_vol_path = f"stage_vol_{unique_hash}.pkl"

    if Path(stage_vol_path).exists():
        # load precalculated stage_vol_table from json file
        print(f"loading stage-volume table from {stage_vol_path}")
        print(f"elevation file: {hand_path}")
        print(f"geometry file: {volume_geometry_path}")
        with open(stage_vol_path, "rb") as f:
            stage_vol_tables = pickle.load(f)
    else:
        # calculate stage_vol_table
        print(f"calculating stage-volume table")
        print(f"elevation file: {hand_path}")
        print(f"geometry file: {volume_geometry_path}")
        stage_vol_tables = get_stage_vol_table(
            geometry_vol, hand_path, cell_area
        )
        # Save dictionary of DataFrames
        with open(stage_vol_path, "wb") as f:
            pickle.dump(stage_vol_tables, f)
        print(f"stage-volume table saved to {stage_vol_path}")

    for idx, row in geometry_vol.iterrows():
        # interpolate h from stage_vol_tables[idx]
        geometry_vol.loc[idx, "H"] = np.interp(
            row["ponded_wat_vol"],
            stage_vol_tables[idx]["vol"],
            stage_vol_tables[idx]["H"],
        )

    return df_float64_to_float32(geometry_vol.drop(columns=["geometry"]))


def get_stage_vol_table(geometry_vol, hand_path, cell_area):
    hand_dict = clip_raster_by_gdf(hand_path, geometry_vol)
    stage_vol_tables = {}
    # wrap for loop in tqdm to show progress bar
    with tqdm(total=len(geometry_vol)) as pbar:
        for geom_idx, geom_row in geometry_vol.iterrows():
            # print(geom_idx)
            hand_clipped = hand_dict[geom_idx][0]
            # vol = geom_row["ponded_wat_vol"]
            stage_vol_table = pd.DataFrame(columns=["H", "vol"])
            # H column ranges from 0 to 20 by 0.1 increments
            stage_vol_table["H"] = np.arange(0, 20.1, 0.1)

            if np.all(np.isnan(hand_clipped)):
                stage_vol_table["vol"] = 0
            else:
                # normalize hand_clipped min value to 0
                elev_min = np.nanmin(hand_clipped)
                hand_clipped = hand_clipped - elev_min
                for h_idx, h in enumerate(stage_vol_table["H"]):
                    # array of inundation depth
                    inun_h = h - hand_clipped
                    inun_h[inun_h < 0] = 0
                    # convert to volume by summing all cells and multiplying by cell area
                    inun_vol = np.nansum(inun_h) * cell_area
                    stage_vol_table.loc[h_idx, "vol"] = inun_vol
            # denormalize H
            stage_vol_table["H"] = stage_vol_table["H"] + elev_min
            stage_vol_tables[geom_idx] = stage_vol_table
            pbar.update(1)

    return stage_vol_tables


def downscale_vol_hand(
    hand_path,
    volume_geometry_path,
    mesh_path,
    ponded_wat_field="ponded_wat",
):
    # read HAND raster's profile
    with rio.open(hand_path) as ds:
        hand_profile = ds.profile
    # volume is sum * xres * yres of ats_pond_raster, m^3
    cell_area = abs(
        hand_profile["transform"].a * hand_profile["transform"].e
    )

    # read geometry within which to spread ATS ponded water
    # could be catchments, mesh cells or groups of mesh cells
    volume_geometry_raw = gpd.read_file(volume_geometry_path)
    # if segment catchments generated from GeoFlood, clean data
    # keep only HYDROID corresponding to largest AreaSqKm
    # first sort dataframe by HYDROID then by AreaSqKm in descending order
    if "HYDROID" and "AreaSqKm" in volume_geometry_raw.columns:
        volume_geometry_sort = volume_geometry_raw.sort_values(
            by=["HYDROID", "AreaSqKm"], ascending=[True, False]
        )
        # group by HYDROID, take first row of each group with highest AreaSqKm
        volume_geometry = (
            volume_geometry_sort.groupby("HYDROID").first().reset_index()
        )
    else:
        volume_geometry = volume_geometry_raw
    del volume_geometry_raw

    # read mesh with ponded water data (ATS output)
    ponded_mesh = gpd.read_file(mesh_path)

    # if volume_geometry_path is the same as mesh_path, calculate volume directly
    # rather than rasterizing the mesh and using zonalstats to
    # sum ponded water in each volume_geometry polygon
    if volume_geometry_path == mesh_path:
        geometry_vol = ponded_mesh.copy()
        del ponded_mesh
        geometry_vol["ponded_wat_vol"] = (
            geometry_vol[ponded_wat_field] * geometry_vol.geometry.area
        )
    else:
        # rasterize ponded water mesh vector geometry
        ponded_raster = rasterize(
            # iterable of (geometry, value) pairs or geometries
            zip(ponded_mesh.geometry, ponded_mesh[ponded_wat_field]),
            out_shape=(hand_profile["height"], hand_profile["width"]),
            dtype=hand_profile["dtype"],
            transform=hand_profile["transform"],
            fill=np.nan,
        )
        del ponded_mesh

        # Sum ponded water volume in each volume_geometry polygon
        stats = zonal_stats(
            volume_geometry,
            ponded_raster,
            affine=hand_profile["transform"],
            stats="sum",
            nodata=np.nan,
            geojson_out=True,
            prefix="ponded_wat_",
        )
        # geodataframe of segment catchment with ponded water volume
        geometry_vol = gpd.GeoDataFrame.from_features(stats)
        geometry_vol["ponded_wat_vol"] = (
            geometry_vol["ponded_wat_sum"] * cell_area
        )

    # save rasterized mesh mapping each geometry to grid cells for jit_inun
    geom_map = rasterize(
        # iterable of (geometry, value) pairs or geometries
        ((geometry, idx) for idx, geometry in enumerate(geometry_vol.geometry)),
        out_shape=(hand_profile["height"], hand_profile["width"]),
        dtype="float32",
        transform=hand_profile["transform"],
        fill=-9999,
    )
    geom_map[geom_map == -9999] = np.nan

    # get dataframe of flood stage for each downscaling geometry
    # this is the height of water needed to fill the geometry to the volume
    # use get_flood_stage_without_table to skip the stage-volume table calculation
    # and directly calculate flood stage
    flood_stage = get_flood_stage(
        geometry_vol,
        volume_geometry_path,
        hand_path,
        cell_area,
        constrained=False,
    )

    # convert flood stage to inundation
    out_profile = hand_profile.copy()
    out_profile.update(compress="lzw", dtype="float32")
    # read HAND raster
    with rio.open(hand_path) as ds:
        hand = ds.read(1)
    hand[hand == hand_profile["nodata"]] = np.nan

    # calculate inundation
    inundated = jit_inun(
        hand,
        geom_map,
        flood_stage.index.to_numpy(),
        flood_stage["H"].to_numpy(),
    )

    return inundated, out_profile


if __name__ == "__main__":
    # run downscale_vol_hand with command line arguments
    # use argparse, print help message
    parser = argparse.ArgumentParser(
        description="Downscale ponded water mesh using HAND raster"
    )

    parser.add_argument(
        "hand_path",
        type=str,
        help="Path to the HAND raster",
    )
    parser.add_argument(
        "volume_geometry_path",
        type=str,
        help="Path to the vector geometry within which to spread ATS ponded water",
    )
    parser.add_argument(
        "mesh_path",
        type=str,
        help="Path to the mesh with ponded water data (ATS output)",
    )
    parser.add_argument(
        "ponded_wat_field",
        type=str,
        default="ponded_wat",
        help="Field name of ponded water data",
    )
    parser.add_argument(
        "out_inun_path",
        type=str,
        help="Path to write the downscaled inundation raster",
    )

    args = parser.parse_args()

    inundated, out_profile = downscale_vol_hand(
        args.hand_path,
        args.volume_geometry_path,
        args.mesh_path,
        args.ponded_wat_field,
    )

    with rio.open(args.out_inun_path, "w", **out_profile) as ds:
        ds.write(inundated, 1)

    print(f"inundation written to {args.out_inun_path}")
