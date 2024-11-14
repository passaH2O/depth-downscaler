Usage
---

See the example Jupyter workflows or use the command line tool. Example data is saved [here](https://utexas.box.com/s/ta31ewmffzged87dv3ec9xt2hb4kao11).

```bash
$ python3 depth_downscaler.py --help
usage: depth_downscaler.py [-h] elev_path volume_geometry_path mesh_path ponded_wat_field out_inun_path

Downscale ponded water mesh using an elevation raster

positional arguments:
  elev_path             Path to the elevation raster (HAND or detrended DEM)
  volume_geometry_path  Path to the vector geometry within which to spread ATS ponded water
  mesh_path             Path to the mesh with ponded water data (ATS output)
  ponded_wat_field      Field name of ponded water data
  out_inun_path         Path to write the downscaled inundation raster

options:
  -h, --help            show this help message and exit
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

