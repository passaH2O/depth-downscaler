#!/usr/bin/env python3

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
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.transform import rowcol
from rasterstats import zonal_stats
from shapely.geometry import mapping
from tqdm import tqdm


def _read_geom(geom_or_path, columns=None):
    """Accept either a GeoDataFrame or a path; return a GeoDataFrame.

    When reading from a path, ``columns`` limits which attribute fields are
    loaded (geometry is always included). Ignored for GeoDataFrame inputs.
    """
    if isinstance(geom_or_path, gpd.GeoDataFrame):
        return geom_or_path
    if columns is not None:
        return gpd.read_file(geom_or_path, columns=columns)
    return gpd.read_file(geom_or_path)


def _same_mesh(a, b) -> bool:
    """True if two inputs refer to the same mesh (by identity or path)."""
    if isinstance(a, gpd.GeoDataFrame) or isinstance(b, gpd.GeoDataFrame):
        return a is b
    return str(a) == str(b)


def load_polygon_ids(path):
    with rio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def rasterize_polygon_ids(polygons, profile, all_touched=True):
    """
    Rasterize polygons to an integer polygon-ID raster.

    Every pixel touched by a polygon (usually mesh cells or river segment catchments)
    is mapped to exactly one polygon; ties among overlapping polygons are
    resolved by iteration order of ``polygons.geometry`` (later geometries win).
    Makes a consistent pixel -> polygon map for both detrending and downscaling.

    Returns
    -------
    ndarray of float32, shape (profile['height'], profile['width'])
        Polygon index (0 .. len(polygons)-1) per pixel, NaN where no polygon overlaps.
    """
    ids = rasterize(
        ((geom, idx) for idx, geom in enumerate(polygons.geometry)),
        out_shape=(profile["height"], profile["width"]),
        dtype="float32",
        transform=profile["transform"],
        fill=-9999,
        all_touched=all_touched,
    )
    ids[ids == -9999] = np.nan
    return ids


def clean_segment_catchments(segment_catchments_path):
    """
    Clean and deduplicate segment catchments generated with pyGeoFlood.
    """
    gdf = _read_geom(segment_catchments_path)
    # when a HYDROID is repeated keep only the one with largest AreaSqKm
    if ("HYDROID" in gdf.columns) and ("AreaSqKm" in gdf.columns):
        gdf = (
            gdf.sort_values(by=["HYDROID", "AreaSqKm"], ascending=[True, False])
               .groupby("HYDROID").first().reset_index()
        )
    return gdf


def write_polygon_ids(polygon_ids, profile, out_path, write_bigtiff=False):
    """Persist a polygon_ids raster to disk that maps pixels to polygons."""
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32", compress="lzw", nodata=-9999,
        tiled=True, blockxsize=512, blockysize=512,
    )
    if write_bigtiff:
        out_profile.update(BIGTIFF="yes")
    out = np.where(np.isnan(polygon_ids), -9999, polygon_ids).astype("float32")
    with rio.open(out_path, "w", **out_profile) as ds:
        ds.write(out, 1)
    print(f"polygon IDs written to {out_path}")
    return out_path


def extract_corners(mesh):
    """
    Build the (N, 3, 3) corners-per-cell array for jit_detrend_all.

    Uses the first 3 exterior coordinates per polygon. Works for triangles
    (all 3 vertices used) and quads (first 3 vertices; fine for planar fit as
    long as not collinear).

    Returns (N, 3, 3) float32 array where [:, :, 0]=x, [:, :, 1]=y, [:, :, 2]=z.
    If the input coordinates are 2D, z is filled with NaN (caller must handle,
    e.g. with use_dem_corner_elev=True).
    """
    N = len(mesh)
    corners = np.full((N, 3, 3), np.nan, dtype=np.float32)
    for i, geom in enumerate(mesh.geometry):
        if geom.geom_type == "MultiPolygon":
            poly = list(geom.geoms)[0]
        elif geom.geom_type == "Polygon":
            poly = geom
        else:
            raise ValueError(f"Cell {i}: geometry must be Polygon or MultiPolygon")
        coords = list(poly.exterior.coords)[:3]
        for k, pt in enumerate(coords):
            corners[i, k, 0] = pt[0]
            corners[i, k, 1] = pt[1]
            if len(pt) >= 3:
                corners[i, k, 2] = pt[2]
    return corners


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


def jit_inun_py(elev, seg_catch, hydroids, stage_m):
    # elev: 2D array
    # seg_catch: 2D array of geometry IDs, same shape as elev
    # hydroids: 1D array of IDs (must be sorted)
    # stage_m: 1D array of water levels aligned with hydroids

    inun = np.empty_like(elev, dtype=np.float32)
    inun.fill(np.nan)

    # simple binary search replacement using numpy indexing
    # (since hydroids are 0..N-1 in your case, this should just be a direct index)
    # If that assumption holds, we can skip search entirely:
    # h = stage_m[int(hydroid)] whenever hydroid is not nan.
    for i in range(elev.shape[0]):
        for j in range(elev.shape[1]):
            hydroid = seg_catch[i, j]
            elev_h = elev[i, j]

            if np.isnan(hydroid):
                continue

            idx = int(hydroid)
            if idx < 0 or idx >= len(stage_m):
                continue

            h = stage_m[idx]
            if h > elev_h:
                inun[i, j] = h - elev_h

    return inun


@jit(nopython=True, parallel=True)
def jit_inun(elev, seg_catch, hydroids, stage_m):
    inun = np.empty_like(elev, dtype=np.float32)
    inun.fill(np.nan)
    for i in prange(elev.shape[0]):
        for j in prange(elev.shape[1]):
            hydroid = seg_catch[i, j]
            elev_h = elev[i, j]
            hydroid_idx = binary_search(hydroids, hydroid)
            if hydroid_idx != -1:
                h = stage_m[hydroid_idx]
            else:
                h = -9999
            if h > elev_h:
                inun[i, j] = h - elev_h
    return inun


def get_file_metadata_hash(filepath):
    # Get file size and modification timestamp
    file_stats = os.stat(filepath)
    metadata_string = f"{file_stats.st_size}_{file_stats.st_mtime}"
    # Generate a hash based on the file metadata
    return hashlib.sha256(metadata_string.encode()).hexdigest()[:8]


def clip_raster_by_gdf(raster_path, gdf, insertion_points=False):
    """
    read raster_path with rasterio
    clip to the geometry of gdf
    store clipped rasters, transforms, and top left corner insertion points
    as tuples in a dictionary with row index as key
    """
    # Open the raster file
    with rio.open(raster_path) as ds:
        full_transform = ds.transform
        out_image_dict = {}
        with tqdm(total=len(gdf), desc="clip elev by geom") as pbar:
            for idx, row in gdf.iterrows():
                # The geometry must be in GeoJSON format
                geom = [mapping(row["geometry"])]
                # Perform the clipping
                out_image, out_transform = mask(ds, geom, crop=True, all_touched=True)
                out_image[out_image == ds.nodata] = np.nan

                if insertion_points:
                    # get insertion point of top left corner of clipped image wrt full image
                    row_insert, col_insert = rowcol(
                        transform=full_transform,
                        xs=out_transform.c,
                        ys=out_transform.f,
                    )
                    # Store the numpy array, transform, and insertion point
                    out_image_dict[idx] = (
                        out_image[0],
                        out_transform,
                        (row_insert, col_insert),
                    )

                else:
                    # Store both the numpy array and its transform
                    out_image_dict[idx] = (out_image[0], out_transform)
                pbar.update(1)

    return out_image_dict


def get_flood_stage(
    geometry_vol,
    volume_geometry_path,
    elev_path,
    cell_area,
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
    elev_path : `str`
        Path to the elevation raster (HAND or detrended DEM).
    cell_area : `float`
        Area of each cell in the elevation raster. Calculated as xres*yres:
        abs(profile["transform"].a * profile["transform"].e)

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
        hash2 = get_file_metadata_hash(elev_path)
        unique_hash = hashlib.sha256((hash1 + hash2).encode()).hexdigest()[:8]
        stage_vol_path = f"stage_vol_{unique_hash}.pkl"

    if Path(stage_vol_path).exists():
        # load precalculated stage_vol_table from json file
        print(f"loading stage-volume table from {stage_vol_path}")
        with open(stage_vol_path, "rb") as f:
            stage_vol_tables = pickle.load(f)
    else:
        # calculate stage_vol_table
        print(f"calculating stage-volume table")
        stage_vol_tables = get_stage_vol_table(geometry_vol, elev_path, cell_area)
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


def get_stage_vol_table(geometry_vol, elev_path, cell_area):
    elev_dict = clip_raster_by_gdf(elev_path, geometry_vol)
    stage_vol_tables = {}
    # wrap for loop in tqdm to show progress bar
    with tqdm(total=len(geometry_vol), desc="calc stage-vol tables") as pbar:
        for geom_idx, geom_row in geometry_vol.iterrows():
            # print(geom_idx)
            elev_clipped = elev_dict[geom_idx][0]
            # vol = geom_row["ponded_wat_vol"]
            stage_vol_table = pd.DataFrame(columns=["H", "vol"])
            # H column ranges from 0 to 20 by 0.1 increments
            stage_vol_table["H"] = np.arange(0, 20.1, 0.1)

            # initialize elev_min to 0
            elev_min = 0
            if np.all(np.isnan(elev_clipped)):
                stage_vol_table["vol"] = 0
                print(f'Warning: all NaN elevation for geometry index {geom_idx}')
            else:
                # normalize elev_clipped min value to 0
                elev_min = np.nanmin(elev_clipped)
                elev_clipped = elev_clipped - elev_min
                for h_idx, h in enumerate(stage_vol_table["H"]):
                    # array of inundation depth
                    inun_h = h - elev_clipped
                    inun_h[inun_h < 0] = 0
                    # convert to volume by summing all cells and multiplying by cell area
                    inun_vol = np.nansum(inun_h) * cell_area
                    stage_vol_table.loc[h_idx, "vol"] = inun_vol
            # denormalize H
            stage_vol_table["H"] = stage_vol_table["H"] + elev_min
            stage_vol_tables[geom_idx] = stage_vol_table
            pbar.update(1)

    return stage_vol_tables


def downscale_vol_elev(
    elev_path,
    volume_geometry_path,
    mesh_path,
    ponded_wat_field="ponded_wat",
    custom_stage_vol_path=None,
):
    # read elev raster's profile
    with rio.open(elev_path) as ds:
        elev_profile = ds.profile
    # volume is sum * xres * yres of ats_pond_raster, m^3
    cell_area = abs(elev_profile["transform"].a * elev_profile["transform"].e)

    # read geometry within which to spread ATS ponded water
    # could be catchments, mesh cells or groups of mesh cells
    volume_geometry_raw = _read_geom(volume_geometry_path)
    # if segment catchments generated from GeoFlood, clean data
    # keep only HYDROID corresponding to largest AreaSqKm
    # first sort dataframe by HYDROID then by AreaSqKm in descending order
    if ("HYDROID" in volume_geometry_raw.columns) and ("AreaSqKm" in volume_geometry_raw.columns):
        volume_geometry_sort = volume_geometry_raw.sort_values(
            by=["HYDROID", "AreaSqKm"], ascending=[True, False]
        )
        # group by HYDROID, take first row of each group with highest AreaSqKm
        volume_geometry = volume_geometry_sort.groupby("HYDROID").first().reset_index()
    else:
        volume_geometry = volume_geometry_raw
    del volume_geometry_raw

    # read mesh with ponded water data (ATS output)
    ponded_mesh = _read_geom(mesh_path)

    # if volume_geometry_path is the same as mesh_path, calculate volume directly
    # rather than rasterizing the mesh and using zonalstats to
    # sum ponded water in each volume_geometry polygon
    if _same_mesh(volume_geometry_path, mesh_path):
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
            out_shape=(elev_profile["height"], elev_profile["width"]),
            dtype=elev_profile["dtype"],
            transform=elev_profile["transform"],
            fill=np.nan,
        )
        del ponded_mesh

        # Sum ponded water volume in each volume_geometry polygon
        stats = zonal_stats(
            volume_geometry,
            ponded_raster,
            affine=elev_profile["transform"],
            stats="sum",
            nodata=np.nan,
            geojson_out=True,
            prefix="ponded_wat_",
        )
        # geodataframe of segment catchment with ponded water volume
        geometry_vol = gpd.GeoDataFrame.from_features(stats)
        geometry_vol["ponded_wat_vol"] = geometry_vol["ponded_wat_sum"] * cell_area

    # save rasterized mesh mapping each geometry to grid cells for jit_inun
    geom_map = rasterize(
        # iterable of (geometry, value) pairs or geometries
        ((geometry, idx) for idx, geometry in enumerate(geometry_vol.geometry)),
        out_shape=(elev_profile["height"], elev_profile["width"]),
        dtype="float32",
        transform=elev_profile["transform"],
        fill=-9999,
        all_touched=True,
    )

    geom_map[geom_map == -9999] = np.nan

    # get dataframe of flood stage for each downscaling geometry
    # this is the height of water needed to fill the geometry to the volume
    # use get_flood_stage_without_table to skip the stage-volume table calculation
    # and directly calculate flood stage
    flood_stage = get_flood_stage(
        geometry_vol,
        volume_geometry_path,
        elev_path,
        cell_area,
        custom_stage_vol_path,
    )

    print("calculating inundation")

    # convert flood stage to inundation
    out_profile = elev_profile.copy()
    out_profile.update(
        compress="lzw",
        dtype="float32",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    # read elev raster
    with rio.open(elev_path) as ds:
        elev = ds.read(1)
    elev[elev == elev_profile["nodata"]] = np.nan

    # calculate inundation
    inundated = jit_inun(
        elev,
        geom_map,
        flood_stage.index.to_numpy(),
        flood_stage["H"].to_numpy(),
    )

    # dissolve the mesh into a single geometry
    outer_union = geometry_vol.geometry.union_all()

    # build mask only for the outer boundary
    mesh_mask = geometry_mask(
        [outer_union],
        transform=elev_profile["transform"],
        out_shape=inundated.shape,
        all_touched=False,  # use center-only test for strict boundary
        invert=True         # keep interior as True
    )

    # apply mask
    inundated[~mesh_mask] = np.nan

    return inundated, out_profile


def detrend_dem(dem_path, mesh_path, out_path, use_dem_corner_elev, write_bigtiff):

    with rio.open(dem_path) as ds:
        dem = ds.read(1)
        dem_profile = ds.profile
    dem[dem == dem_profile["nodata"]] = np.nan

    mesh = gpd.read_file(mesh_path)
    quad_clipped_dem_dict = clip_raster_by_gdf(dem_path, mesh, insertion_points=True)

    quad_list = []
    insertion_list = []

    with tqdm(total=len(quad_clipped_dem_dict), desc="detrend cells") as pbar:
        for idx in quad_clipped_dem_dict.keys():
            quad, transform, insert_rowcol = quad_clipped_dem_dict[idx]
            quad_detrended = detrend_quad(
                quad, transform, mesh.loc[idx, "geometry"], use_dem_corner_elev
            )
            quad_list.append(quad_detrended)
            insertion_list.append(insert_rowcol)
            pbar.update(1)

    print("writing DEM")

    detrended_dem = stitch_arrays(quad_list, insertion_list, dem.shape)

    out_profile = dem_profile.copy()
    out_profile.update(
        dtype="float32",
        compress="lzw",
        nodata=-999999,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    # force BIGTIFF if < 4 GB, not handled automatically with compressed GeoTIFFs
    if write_bigtiff:
        out_profile.update(BIGTIFF="yes")

    with rio.open(out_path, "w", **out_profile) as ds:
        ds.write(detrended_dem, 1)

    print(f"Detrended DEM written to {out_path}")


def stitch_arrays(array_list, insertion_point_list, out_shape):
    stitched = np.empty(out_shape, "float32")
    stitched.fill(np.nan)
    with tqdm(total=len(array_list), desc="stitch detrended cells") as pbar:
        for array, (row_start, col_start) in zip(array_list, insertion_point_list):
            height, width = array.shape
            row_stop = row_start + height
            col_stop = col_start + width
            target_region = stitched[row_start:row_stop, col_start:col_stop]
            valid_mask = ~np.isnan(array)
            target_region[valid_mask] = array[valid_mask]
            pbar.update(1)
    return stitched


# get nearest DEM z elevation at x, y coords of a quad corner
@jit(nopython=True)
def jit_get_nearest_elevation(dem_array, transform, x, y, max_radius):
    """
    Given a DEM array, its affine transform, and a point (x, y),
    return the elevation of the nearest non-NaN pixel within an expanding window.

    The transform is expected to be a 6-tuple where:
      transform[0] = pixel width (t0)
      transform[2] = x offset (t2)
      transform[4] = pixel height (t4)   (often negative)
      transform[5] = y offset (t5)

    The conversion used is the reverse of:
        x = t2 + t0 * (j + 0.5)
        y = t5 + t4 * (i + 0.5)
    """
    t0 = transform[0]
    t2 = transform[2]
    t4 = transform[4]
    t5 = transform[5]

    # Reverse the transformation to get fractional column and row indices.
    # Note: the formula is:
    #   j = (x - t2) / t0 - 0.5
    #   i = (y - t5) / t4 - 0.5
    col_f = (x - t2) / t0 - 0.5
    row_f = (y - t5) / t4 - 0.5

    # Round to get the nearest integer indices.
    col = int(round(col_f))
    row = int(round(row_f))

    nrows = dem_array.shape[0]
    ncols = dem_array.shape[1]

    # If out-of-bounds, return NaN.
    if row < 0 or row >= nrows or col < 0 or col >= ncols:
        return np.nan

    base_val = dem_array[row, col]
    if not np.isnan(base_val):
        return base_val

    # Search within a window (up to max_radius) for the nearest valid elevation.
    best_val = np.nan
    best_dist = 1e10  # large number
    row_min = row - max_radius if row - max_radius >= 0 else 0
    row_max = row + max_radius if row + max_radius < nrows else nrows - 1
    col_min = col - max_radius if col - max_radius >= 0 else 0
    col_max = col + max_radius if col + max_radius < ncols else ncols - 1
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            val = dem_array[r, c]
            if not np.isnan(val):
                # Compute Euclidean distance in pixel space.
                d = ((r - row) ** 2 + (c - col) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_val = val
    return best_val


@jit(nopython=True)
def jit_detrend_quad(quad, transform, corners, use_dem_corner_elev, subtract_min=True):
    """
    Detrend a quad using a plane fitted to the exterior of a polygon.

    Parameters:
    - quad: 2D numpy array representing the quad to be detrended.
    - transform: Tuple of rasterio affine transformation coefficients.
    - corners: 2D numpy array of shape (3, 3), where each row represents the x, y, z
      coordinates of a corner of the polygon.
    - use_dem_corner_elev: Boolean indicating whether to use nearest DEM elevation
      to each corner rather than the z value of the quad's corners.
    """

    if use_dem_corner_elev:
        # rather than use z value of quad's corners, use nearest DEM elevation
        # to each corner (seems less accurate than using quad's z values)
        max_radius = 4  # maximum search radius in pixels
        # Update each corner's z with the nearest non-nodata value from the DEM (quad)
        for k in range(3):
            x = corners[k, 0]
            y = corners[k, 1]
            corners[k, 2] = jit_get_nearest_elevation(quad, transform, x, y, max_radius)

    else:
        # Average elevation of the corners
        z_avg = np.mean(corners[:, 2])

    # Plane coefficients, assuming elevation = Ax + By + C
    A_matrix = np.column_stack((corners[:, 0], corners[:, 1], np.ones(3)))
    z_vector = corners[:, 2]
    A, B, C = np.linalg.solve(A_matrix, z_vector)

    # Initialize adjustment map array
    adjustment_map = np.zeros(quad.shape, dtype=np.float32)
    t2 = transform[2]
    t0 = transform[0]
    t5 = transform[5]
    t4 = transform[4]
    for i in range(quad.shape[0]):
        for j in range(quad.shape[1]):
            # Manual conversion of array row, col to x, y using affine transform
            # Offset by half the cell size for center offset
            x = t2 + t0 * (j + 0.5)  # A + B*col + 0.5*B
            y = t5 + t4 * (i + 0.5)  # D + F*row + 0.5*F
            # Expected elevation on the plane
            z_expected = A * x + B * y + C
            # Adjustment needed to match the average elevation
            adjustment_map[i, j] = z_avg - z_expected
    # adjust the quad by the adjustment map
    detrended = quad + adjustment_map
    # set detrended quad's minimum value to 0
    if subtract_min:
        detrended = detrended - np.nanmin(detrended)

    return detrended


def detrend_quad(quad, transform, polygon, use_dem_corner_elev, subtract_min=True):
    """
    Detrend a quad using a plane fitted to the exterior of a polygon.

    Parameters
    ----------
    quad : np.ndarray
        2D numpy array representing the quad to be detrended.
    transform : tuple
        Tuple of rasterio affine transformation coefficients.
    polygon : shapely.geometry.Polygon
        Polygon used to fit the plane for detrending.

    Returns
    -------
    quad_detrended : np.ndarray
        Detrended quad.
    """
    if polygon.geom_type == "MultiPolygon":
        exterior_coords = [list(x.exterior.coords) for x in polygon.geoms][0]
    elif polygon.geom_type == "Polygon":
        exterior_coords = list(polygon.exterior.coords)
    else:
        raise ValueError("Geometry must be a Polygon or MultiPolygon")
    corners = np.array(exterior_coords[:3])
    quad_detrended = jit_detrend_quad(
        quad, transform, corners, use_dem_corner_elev, subtract_min
    )
    quad_detrended[np.isnan(quad)] = np.nan
    return quad_detrended


def downscale_and_write(
    elev_path,
    volume_geometry_path,
    mesh_path,
    out_inun_path,
    ponded_wat_field="ponded_wat",
    custom_stage_vol_path=None,
    write_bigtiff=False,
):
    """Run downscale_vol_elev and write the inundation raster."""
    inundated, out_profile = downscale_vol_elev(
        str(elev_path),
        volume_geometry_path,
        mesh_path,
        ponded_wat_field=ponded_wat_field,
        custom_stage_vol_path=(
            str(custom_stage_vol_path) if custom_stage_vol_path is not None else None
        ),
    )
    if write_bigtiff:
        out_profile.update(BIGTIFF="yes")
    with rio.open(out_inun_path, "w", **out_profile) as ds:
        ds.write(inundated, 1)
    print(f"inundation written to {out_inun_path}", flush=True)
    return out_inun_path


def select_fluvial_mesh(
    mesh,
    segments,
    inundation_path,
    distance=-1,
    fraction=0.33,
):
    """
    Select mesh cells likely to have fluvial inundation.

    Cells are included if they (a) touch a stream segment, or (b) are within
    ``distance`` meters of a stream AND have an inundated-fraction >=
    ``fraction``. Only the contiguous component containing stream-touching
    cells is returned.

    Parameters
    ----------
    mesh : str, Path, or GeoDataFrame
        Mesh cells. Gets an ``element_ID`` column if absent.
    segments : str, Path, or GeoDataFrame
        Stream segments.
    inundation_path : str or Path
        Downscaled inundation raster.
    distance : float
        Max distance from streams in meters. -1 disables the distance filter.
    fraction : float
        Minimum fraction of inundated pixels per cell.

    Returns
    -------
    GeoDataFrame
        Contiguous fluvial cells with ``element_ID`` and ``geometry``.
    """
    mesh = _read_geom(mesh, columns=["element_ID", "geometry"])
    if "element_ID" not in mesh.columns:
        mesh = mesh.copy()
        mesh["element_ID"] = mesh.index
    mesh = mesh[["element_ID", "geometry"]]

    segments = _read_geom(segments)
    segments_union = segments.union_all()

    print("Selecting mesh cells that touch streams...")
    touches_streams_mask = mesh.intersects(segments_union)
    mesh_touches_streams = mesh[touches_streams_mask].copy()
    print(f"  {len(mesh_touches_streams)} cells directly touch streams")

    # determine which cells to check for inundation
    if distance == -1:
        print("No distance limit, checking all mesh cells for inundation...")
        mesh_candidates = mesh.copy()
    else:
        # select mesh cells within buffer distance of streams
        print(f"Buffering streams by {distance} m...")
        stream_buffer = segments_union.buffer(distance)
        near_streams_mask = mesh.intersects(stream_buffer)
        mesh_candidates = mesh[near_streams_mask].copy()
        print(f"  {len(mesh_candidates)} cells within {distance} m of streams")

    print(f"Calculating inundation fraction from {inundation_path}...")
    with rio.open(inundation_path) as src:
        affine = src.transform
        nodata = src.nodata
        ones_shape = (src.height, src.width)

    # get count of inundated pixels per cell
    stats = zonal_stats(
        mesh_candidates,
        str(inundation_path),
        affine=affine,
        stats=["count"],
        nodata=nodata,
        all_touched=True,
    )
    inundated_counts = [s["count"] if s["count"] is not None else 0 for s in stats]

    # get count of total pixels per cell
    ones = np.ones(ones_shape, dtype=np.float32)
    stats_total = zonal_stats(
        mesh_candidates,
        ones,
        affine=affine,
        stats=["count"],
        nodata=-999,
        all_touched=True,
    )
    total_counts = [s["count"] if s["count"] is not None else 1 for s in stats_total]

    # calculate inundation fraction
    mesh_candidates["inundated_pixels"] = inundated_counts
    mesh_candidates["total_pixels"] = total_counts
    mesh_candidates["inundated_fraction"] = (
        mesh_candidates["inundated_pixels"] / mesh_candidates["total_pixels"]
    )

    # filter by inundation fraction
    fluvial_mask = mesh_candidates["inundated_fraction"] >= fraction
    mesh_inundated = mesh_candidates[fluvial_mask].copy()
    print(f"  {len(mesh_inundated)} cells with >= {fraction:.0%} inundated")

    # combine stream-touching and inundated candidates
    fluvial_ids = set(mesh_touches_streams["element_ID"]).union(
        set(mesh_inundated["element_ID"])
    )
    mesh_fluvial = mesh[mesh["element_ID"].isin(fluvial_ids)].copy()
    print(f"  {len(mesh_fluvial)} candidate fluvial cells (streams + inundated)")

    print("Filtering to contiguous region touching streams...")
    dissolved = mesh_fluvial.union_all()
    # handle single polygon and multipolygon cases
    if dissolved.geom_type == "MultiPolygon":
        components = list(dissolved.geoms)
    else:
        components = [dissolved]

    # keep only contiguous components that touch stream-touching cells
    stream_touching_union = mesh_touches_streams.union_all()
    valid_components = []
    for comp in components:
        if comp.intersects(stream_touching_union):
            valid_components.append(comp)
    if valid_components:
        valid_region = gpd.GeoSeries(valid_components).union_all()
        contiguous_mask = mesh_fluvial.intersects(valid_region)
        mesh_fluvial = mesh_fluvial[contiguous_mask].copy()

    print(f"  {len(mesh_fluvial)} contiguous fluvial cells")
    return mesh_fluvial[["element_ID", "geometry"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tools for downscaling ponded water outputs"
    )

    subparsers = parser.add_subparsers(
        title="Subcommands",
        description="Available subcommands",
        dest="command",
        required=True,
    )

    # Subcommand: downscale
    downscale_parser = subparsers.add_parser(
        "downscale",
        help="Downscale ponded water polygon mesh using an elevation raster",
    )
    downscale_parser.add_argument(
        "-e",
        "--elev_path",
        type=str,
        required=True,
        help="Path to the elevation raster (HAND or detrended DEM)",
    )
    downscale_parser.add_argument(
        "-v",
        "--volume_geometry_path",
        type=str,
        required=True,
        help="Path to the vector geometry within which to spread ATS ponded water. Likely stream segment catchments or mesh of triangles/quads.",
    )
    downscale_parser.add_argument(
        "-m",
        "--mesh_path",
        type=str,
        required=True,
        help="Path to the mesh with ponded water data (ATS output)",
    )
    downscale_parser.add_argument(
        "-o",
        "--out_inun_path",
        type=str,
        required=True,
        help="Path to write the downscaled inundation raster",
    )
    downscale_parser.add_argument(
        "-p",
        "--ponded_wat_field",
        type=str,
        default="ponded_wat",
        help="(Optional) Mesh field name of ponded water data (default: ponded_wat)",
    )
    downscale_parser.add_argument(
        "-w",
        "--write_bigtiff",
        action="store_true",
        help="(Optional) Write output in BigTIFF format (default: disabled)",
    )
    downscale_parser.add_argument(
        "-c",
        "--custom_stage_vol_path",
        type=str,
        default=None,
        help="(Optional) Path to .pkl file with previously calculated stage-volume tables (default: None). If not provided, looks for existing stage-vol table, otherwise calculates and writes new one.",
    )

    # Subcommand: detrend DEM
    detrend_parser = subparsers.add_parser(
        "detrend",
        help="Detrend a DEM by its slope at each shape in a polygonal geometry",
    )
    detrend_parser.add_argument(
        "-d",
        "--dem_path",
        type=str,
        required=True,
        help="Path to the DEM raster",
    )
    detrend_parser.add_argument(
        "-g",
        "--geometry_path",
        type=str,
        required=True,
        help="Path to the vector geometry used to detrend DEM. Should consist of triangles and quads.",
    )
    detrend_parser.add_argument(
        "-o",
        "--out_detrend_path",
        type=str,
        required=True,
        help="Path to write the detrended DEM",
    )
    detrend_parser.add_argument(
        "--use_dem_corner_elev",
        action="store_true",
        help="(Optional) Use nearest DEM elevation to geometry corner rather than geometry's z elevation (default: disabled)",
    )
    detrend_parser.add_argument(
        "-w",
        "--write_bigtiff",
        action="store_true",
        help="(Optional) Write output in BigTIFF format (default: disabled)",
    )
    args = parser.parse_args()

    if args.command == "downscale":
        inundated, out_profile = downscale_vol_elev(
            args.elev_path,
            args.volume_geometry_path,
            args.mesh_path,
            ponded_wat_field=args.ponded_wat_field,
            custom_stage_vol_path=args.custom_stage_vol_path,
        )

        # force BIGTIFF if < 4 GB, not handled automatically with compressed GeoTIFFs
        if args.write_bigtiff:
            out_profile.update(BIGTIFF="yes")

        with rio.open(args.out_inun_path, "w", **out_profile) as ds:
            ds.write(inundated, 1)
        print(f"inundation written to {args.out_inun_path}")

    elif args.command == "detrend":
        detrend_dem(
            args.dem_path,
            args.geometry_path,
            args.out_detrend_path,
            args.use_dem_corner_elev,
            args.write_bigtiff,
        )
