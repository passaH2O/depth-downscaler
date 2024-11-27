Usage
---

See the example Jupyter workflows or use the command line tool. Example data is saved [here](https://utexas.box.com/s/ta31ewmffzged87dv3ec9xt2hb4kao11).

```bash
$ python3 depth_downscaler.py --help
usage: depth_downscaler.py [-h] {downscale,detrend} ...

Tools for downscaling ponded water outputs

options:
  -h, --help           show this help message and exit

Subcommands:
  Available subcommands

  {downscale,detrend}
    downscale          Downscale ponded water polygon mesh using an elevation raster
    detrend            Detrend a DEM by its slope at each shape in a polygonal geometry



$ python3 depth_downscaler.py downscale --help
usage: depth_downscaler.py downscale [-h] -e ELEV_PATH -v VOLUME_GEOMETRY_PATH -m MESH_PATH [-p PONDED_WAT_FIELD] -o
                                     OUT_INUN_PATH

options:
  -h, --help            show this help message and exit
  -e ELEV_PATH, --elev_path ELEV_PATH
                        Path to the elevation raster (HAND or detrended DEM)
  -v VOLUME_GEOMETRY_PATH, --volume_geometry_path VOLUME_GEOMETRY_PATH
                        Path to the vector geometry within which to spread ATS ponded water
  -m MESH_PATH, --mesh_path MESH_PATH
                        Path to the mesh with ponded water data (ATS output)
  -p PONDED_WAT_FIELD, --ponded_wat_field PONDED_WAT_FIELD
                        Field name of ponded water data (default: ponded_wat)
  -o OUT_INUN_PATH, --out_inun_path OUT_INUN_PATH
                        Path to write the downscaled inundation raster



$ python3 depth_downscaler.py detrend --help
usage: depth_downscaler.py detrend [-h] -d DEM_PATH -g GEOMETRY_PATH -o OUT_DETREND_PATH

options:
  -h, --help            show this help message and exit
  -d DEM_PATH, --dem_path DEM_PATH
                        Path to the DEM raster
  -g GEOMETRY_PATH, --geometry_path GEOMETRY_PATH
                        Path to the vector geometry use to detrend DEM
  -o OUT_DETREND_PATH, --out_detrend_path OUT_DETREND_PATH
                        Path to write the detrended DEM
```

Python3 dependencies:
---
- geopandas
- numba
- numpy
- pandas
- rasterio
- rasterstats
- shapely
- tqdm

