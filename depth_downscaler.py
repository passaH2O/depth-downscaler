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


def get_flood_stage(
    polygons_vol,
    downscaling_polygons_path,
    elev_path,
    cell_area,
    custom_stage_vol_path=None,
    polygon_ids=None,
):
    """
    Calculate the flood stage for each downscaling polygon.

    Parameters
    ----------
    polygons_vol : `geopandas.GeoDataFrame`
        GeoDataFrame of downscaling polygons with ponded water volume.
        Must have a "pw_vol" column.
    downscaling_polygons_path : `str`
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
        hash1 = get_file_metadata_hash(downscaling_polygons_path)
        hash2 = get_file_metadata_hash(elev_path)
        unique_hash = hashlib.sha256((hash1 + hash2).encode()).hexdigest()[:8]
        stage_vol_dir = Path(downscaling_polygons_path).parent
        stage_vol_path = str(stage_vol_dir / f"stage_vol_{unique_hash}.pkl")

    if Path(stage_vol_path).exists():
        # load precalculated stage_vol_table from json file
        print(f"loading stage-volume table from {stage_vol_path}")
        with open(stage_vol_path, "rb") as f:
            stage_vol_tables = pickle.load(f)
        if len(stage_vol_tables) != len(polygons_vol):
            raise ValueError(
                f"stage-volume table has {len(stage_vol_tables)} entries "
                f"but polygons_vol has {len(polygons_vol)}. Stale pickle?"
            )
    else:
        # calculate stage_vol_table
        print(f"calculating stage-volume table")
        stage_vol_tables = get_stage_vol_table_jit(polygons_vol, elev_path, cell_area, polygon_ids)
        # Save dictionary of DataFrames
        with open(stage_vol_path, "wb") as f:
            pickle.dump(stage_vol_tables, f)
        print(f"stage-volume table saved to {stage_vol_path}")

    for idx, row in polygons_vol.iterrows():
        # interpolate h from stage_vol_tables[idx]
        polygons_vol.loc[idx, "H"] = np.interp(
            row["pw_vol"],
            stage_vol_tables[idx]["vol"],
            stage_vol_tables[idx]["H"],
        )

    return df_float64_to_float32(polygons_vol.drop(columns=["geometry"]))


@jit(nopython=True)
def jit_stage_vol(elev, polygon_ids, n_polygons, cell_area, H_values):
    """
    Build per-polygon volume curves in one pass.

    Parameters
    ----------
    elev : (H, W) float32
        Elevation raster, NaN for nodata.
    polygon_ids : (H, W) float32
        Pixel -> polygon iloc assignment, NaN for unowned pixels.
    n_polygons : int
        Number of polygons (len(polygons_vol)).
    cell_area : float
        Pixel area in map units.
    H_values : (n_H,) float32
        Candidate water heights above each polygon's local minimum.

    Returns
    -------
    vols : (n_polygons, n_H) float32
        Volume held in each polygon at each H, before adding elev_min back.
    elev_mins : (n_polygons,) float32
        Per-polygon minimum elevation (+inf where the polygon has no valid pixels).
    """
    n_H = H_values.shape[0]
    H_pix, W_pix = elev.shape

    # First pass: per-polygon minimum elevation
    elev_mins = np.full(n_polygons, np.inf, dtype=np.float32)
    for r in range(H_pix):
        for c in range(W_pix):
            pid_f = polygon_ids[r, c]
            if np.isnan(pid_f):
                continue
            z = elev[r, c]
            if np.isnan(z):
                continue
            pid = int(pid_f)
            if z < elev_mins[pid]:
                elev_mins[pid] = z

    # Second pass: accumulate volume contributions at each H
    vols = np.zeros((n_polygons, n_H), dtype=np.float64)  # first save as depth, convert to volume before return
    for r in range(H_pix):
        for c in range(W_pix):
            pid_f = polygon_ids[r, c]
            if np.isnan(pid_f):
                continue
            z = elev[r, c]
            if np.isnan(z):
                continue
            pid = int(pid_f)
            z_shifted = z - elev_mins[pid]
            # sum each pixel's depth given H and add to containing polygon's total volume
            for h_idx in range(n_H):
                h = H_values[h_idx]
                if h > z_shifted:
                    vols[pid, h_idx] += (h - z_shifted)

    # convert depths to volumes
    vols *= cell_area
    return vols.astype(np.float32), elev_mins


def get_stage_vol_table_jit(polygons_vol, elev_path, cell_area, polygon_ids):
    """Build per-polygon stage-volume tables in one pass."""
    with rio.open(elev_path) as ds:
        elev = ds.read(1).astype("float32")
        nodata = ds.nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan

    H_values = np.arange(0, 20.1, 0.1, dtype=np.float32)
    vols, elev_mins = jit_stage_vol(
        elev, polygon_ids, len(polygons_vol), cell_area, H_values
    )

    stage_vol_tables = {}
    for iloc, (geom_idx, _) in enumerate(polygons_vol.iterrows()):
        z_min = elev_mins[iloc]
        if not np.isfinite(z_min):
            table = pd.DataFrame({"H": H_values, "vol": np.zeros_like(H_values)})
            print(f"Warning: all NaN elevation for geometry index {geom_idx}")
        else:
            table = pd.DataFrame({"H": H_values + z_min, "vol": vols[iloc]})
        stage_vol_tables[geom_idx] = table
    return stage_vol_tables


def downscale(
    elev_path,
    downscaling_polygons_path,
    pw_mesh_path,
    out_inun_path,
    pw_field="pw",
    custom_stage_vol_path=None,
    polygon_ids_path=None,
    write_bigtiff=False,
):
    # read elev raster's profile
    with rio.open(elev_path) as ds:
        elev_profile = ds.profile
    # volume is sum * xres * yres of ats_pond_raster, m^3
    cell_area = abs(elev_profile["transform"].a * elev_profile["transform"].e)

    # read geometry within which to spread ATS ponded water
    # could be catchments, mesh cells or groups of mesh cells
    polygons = clean_segment_catchments(downscaling_polygons_path)

    # read mesh with ponded water data (ATS output)
    pw_mesh = _read_geom(pw_mesh_path)

    # if downscaling_polygons_path is the same as pw_mesh_path, calculate volume directly
    # rather than rasterizing the mesh and using zonalstats to
    # sum ponded water in each downscaling polygon
    if _same_mesh(downscaling_polygons_path, pw_mesh_path):
        polygons_vol = pw_mesh.copy()
        del pw_mesh
        polygons_vol["pw_vol"] = (
            polygons_vol[pw_field] * polygons_vol.geometry.area
        )
    else:
        # rasterize ponded water mesh vector geometry
        ponded_raster = rasterize(
            # iterable of (geometry, value) pairs or geometries
            zip(pw_mesh.geometry, pw_mesh[pw_field]),
            out_shape=(elev_profile["height"], elev_profile["width"]),
            dtype=elev_profile["dtype"],
            transform=elev_profile["transform"],
            fill=np.nan,
        )
        del pw_mesh

        # Sum ponded water volume in each downscaling polygon
        stats = zonal_stats(
            polygons,
            ponded_raster,
            affine=elev_profile["transform"],
            stats="sum",
            nodata=np.nan,
            geojson_out=True,
            prefix="pw_",
        )
        # geodataframe of downscaling polygons with ponded water volume
        polygons_vol = gpd.GeoDataFrame.from_features(stats)
        polygons_vol["pw_vol"] = polygons_vol["pw_sum"] * cell_area

    # Load or calculate the raster mapping pixels to their containing polygons
    if polygon_ids_path is not None:
        polygon_ids = load_polygon_ids(polygon_ids_path)
    else:
        polygon_ids = rasterize_polygon_ids(polygons_vol, elev_profile)

    # get dataframe of flood stage for each downscaling polygon
    # this is the height of water needed to fill the polygon to the volume
    flood_stage = get_flood_stage(
        polygons_vol,
        downscaling_polygons_path,
        elev_path,
        cell_area,
        custom_stage_vol_path,
        polygon_ids=polygon_ids,
    )

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
        polygon_ids,
        flood_stage.index.to_numpy(),
        flood_stage["H"].to_numpy(),
    )

    if write_bigtiff:
        out_profile.update(BIGTIFF="yes")
    with rio.open(out_inun_path, "w", **out_profile) as ds:
        ds.write(inundated, 1)
    print(f"inundation written to {out_inun_path}", flush=True)


def detrend(
    dem_path,
    mesh_path,
    out_detrend_path,
    use_dem_corner_elev=False,
    write_bigtiff=False,
    out_cell_ids_path=None,
    subtract_min=True,
):
    """
    Detrend a DEM per mesh cell, using a single pass jit loop driven by a
    canonical cell-ID raster.

    Writes the detrended DEM to ``out_detrend_path`` and, unless ``out_cell_ids_path`` is
    False, writes the cell_ids raster alongside it (default: same path with
    ``_cellids.tif`` suffix). Pass the cell_ids path to ``downscale``
    to guarantee detrend/downscale agree on pixel-to-cell ownership.
    """
    with rio.open(dem_path) as ds:
        dem = ds.read(1).astype("float32")
        dem_profile = ds.profile
    dem[dem == dem_profile["nodata"]] = np.nan

    mesh = gpd.read_file(mesh_path)

    # Canonical pixel -> cell assignment
    cell_ids = rasterize_polygon_ids(mesh, dem_profile)

    # Per-cell corner (x, y, z) arrays for jit consumption
    corners = extract_corners(mesh)

    # Affine tuple for jit
    t = dem_profile["transform"]
    transform = np.array([t.a, t.b, t.c, t.d, t.e, t.f], dtype=np.float64)

    # Dummy array to satisfy the jit signature (unused in current implementation)
    corner_valid_count = np.ones(len(mesh), dtype=np.int32)

    detrended_dem = jit_detrend_all(
        dem,
        cell_ids,
        corners,
        corner_valid_count,
        bool(use_dem_corner_elev),
        transform,
        bool(subtract_min),
    )

    out_profile = dem_profile.copy()
    out_profile.update(
        dtype="float32",
        compress="lzw",
        nodata=-999999,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    if write_bigtiff:
        out_profile.update(BIGTIFF="yes")

    with rio.open(out_detrend_path, "w", **out_profile) as ds:
        ds.write(detrended_dem, 1)
    print(f"Detrended DEM written to {out_detrend_path}")

    # Write cell_ids raster unless explicitly disabled
    if out_cell_ids_path is not False:
        if out_cell_ids_path is None:
            out_cell_ids_path = str(out_detrend_path).replace(".tif", "_cellids.tif")
        write_polygon_ids(cell_ids, dem_profile, out_cell_ids_path, write_bigtiff)


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


@jit(nopython=True, parallel=True)
def jit_detrend_all(
    dem,
    cell_ids,
    corners_by_cell,
    corner_valid_count,
    use_dem_corner_elev_z,
    transform,
    subtract_min,
):
    """
    Detrend every pixel of a DEM in one pass using a per-cell plane fit.

    Parameters
    ----------
    dem : 2D float32 array
        DEM values (NaN = nodata).
    cell_ids : 2D float32 array
        Cell index per pixel (NaN where unowned).
    corners_by_cell : (N, 3, 3) float32
        First 3 corner coords per cell: [:, :, 0]=x, [:, :, 1]=y, [:, :, 2]=z.
    corner_valid_count : (N,) int32
        1 if the cell has valid corner z values, 0 if they need DEM lookup.
    use_dem_corner_elev_z : bool
        Whether to look up DEM elevation at each corner instead of using the
        polygon's z coordinate.
    transform : (6,) float64
        rasterio affine tuple (a, b, c, d, e, f).
    subtract_min : bool
        If True, subtract the per-cell minimum so detrended starts at 0.

    Returns
    -------
    2D float32 array of detrended values (NaN where unowned or DEM nodata).
    """
    H, W = dem.shape
    N = corners_by_cell.shape[0]

    # optionally override mesh cell corner z values using nearest DEM elevation
    # (seems better to use cell corner z vals)
    if use_dem_corner_elev_z:
        for n in range(N):
            for k in range(3):
                x_k = corners_by_cell[n, k, 0]
                y_k = corners_by_cell[n, k, 1]
                corners_by_cell[n, k, 2] = jit_get_nearest_elevation(
                    dem, transform, x_k, y_k, 4
                )

    # Precompute plane coefficients (A, B, C) for each cell: z = Ax + By + C
    coeffs = np.empty((N, 3), dtype=np.float64)
    z_avg = np.empty(N, dtype=np.float32)
    valid_plane = np.zeros(N, dtype=np.int32)

    for n in range(N):
        x0 = corners_by_cell[n, 0, 0]; y0 = corners_by_cell[n, 0, 1]; z0 = corners_by_cell[n, 0, 2]
        x1 = corners_by_cell[n, 1, 0]; y1 = corners_by_cell[n, 1, 1]; z1 = corners_by_cell[n, 1, 2]
        x2 = corners_by_cell[n, 2, 0]; y2 = corners_by_cell[n, 2, 1]; z2 = corners_by_cell[n, 2, 2]

        # Skip if any corner z is NaN (could not look up)
        if np.isnan(z0) or np.isnan(z1) or np.isnan(z2):
            continue

        A_mat = np.empty((3, 3), dtype=np.float64)
        A_mat[0, 0] = x0; A_mat[0, 1] = y0; A_mat[0, 2] = 1.0
        A_mat[1, 0] = x1; A_mat[1, 1] = y1; A_mat[1, 2] = 1.0
        A_mat[2, 0] = x2; A_mat[2, 1] = y2; A_mat[2, 2] = 1.0
        z_vec = np.empty(3, dtype=np.float64)
        z_vec[0] = z0; z_vec[1] = z1; z_vec[2] = z2
        coeffs[n] = np.linalg.solve(A_mat, z_vec)
        z_avg[n] = (z0 + z1 + z2) / 3.0
        valid_plane[n] = 1

    t0 = transform[0]; t2 = transform[2]; t4 = transform[4]; t5 = transform[5]

    # Pass 1: compute detrended values
    out = np.full((H, W), np.nan, dtype=np.float32)
    for r in prange(H):
        for c in range(W):
            cid_f = cell_ids[r, c]
            if np.isnan(cid_f):
                continue
            cid = int(cid_f)  # mesh cell ID assigned to fine pixel (r, c)
            if valid_plane[cid] == 0:
                continue
            z_dem = dem[r, c]
            if np.isnan(z_dem):
                continue
            # convert array rowcol to projected xy using affine transform
            # 0.5 offset is added to get pixel center coordinates
            x = t2 + t0 * (c + 0.5)
            y = t5 + t4 * (r + 0.5)
            # z_plane is adjustment to apply to the DEM pixel to put it on the cell's plane fit
            z_plane = coeffs[cid, 0] * x + coeffs[cid, 1] * y + coeffs[cid, 2]
            out[r, c] = z_dem + (z_avg[cid] - z_plane)

    # Pass 2 (optional): subtract per-cell minimum
    if subtract_min:
        cell_mins = np.full(N, np.inf, dtype=np.float32)
        for r in range(H):
            for c in range(W):
                cid_f = cell_ids[r, c]
                if np.isnan(cid_f):
                    continue
                v = out[r, c]
                if np.isnan(v):
                    continue
                cid = int(cid_f)
                if v < cell_mins[cid]:
                    cell_mins[cid] = v
        for r in prange(H):
            for c in range(W):
                cid_f = cell_ids[r, c]
                if np.isnan(cid_f):
                    continue
                cid = int(cid_f)
                if cell_mins[cid] < np.inf:
                    v = out[r, c]
                    if not np.isnan(v):
                        out[r, c] = v - cell_mins[cid]

    return out


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
        downscale(
            elev_path=args.elev_path,
            downscaling_polygons_path=args.volume_geometry_path,
            pw_mesh_path=args.mesh_path,
            out_inun_path=args.out_inun_path,
            pw_field=args.ponded_wat_field,
            custom_stage_vol_path=args.custom_stage_vol_path,
            write_bigtiff=args.write_bigtiff,
        )

    elif args.command == "detrend":
        detrend(
            args.dem_path,
            args.geometry_path,
            args.out_detrend_path,
            args.use_dem_corner_elev,
            args.write_bigtiff,
        )
