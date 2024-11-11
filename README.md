Usage
---

See the example Jupyter workflows or use the command line tool:

```bash
$ python3 depth_downscaler.py --help
usage: depth_downscaler.py [-h] hand_path volume_geometry_path mesh_path ponded_wat_field out_inun_path

Downscale ponded water mesh using HAND raster

positional arguments:
  hand_path             Path to the HAND raster
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
