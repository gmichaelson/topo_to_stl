import numpy as np
import rasterio
import os

# Only use the new ERDAS Imagine DEM for Nevada
img_file = "nv_dem.img"  # Official Nevada DEM
with rasterio.open(img_file) as src:
    arr = src.read(1).astype(np.float32)
    profile = src.profile
    crs = src.crs
    nodata = src.nodata
    # Clean where nodata
    mask = (arr == nodata) | np.isnan(arr)
    valid_min = np.nanmin(arr[~mask])
    arr[mask] = valid_min
    sample_factor = 8  # Reduce data for mesh
    arr_ds = arr[::sample_factor, ::sample_factor]
    min_elev = float(np.nanmin(arr_ds))
    max_elev = float(np.nanmax(arr_ds))
    mean_elev = float(np.nanmean(arr_ds))
    n_nodata = int(np.sum(np.isnan(arr_ds)))
    arr_norm = (arr_ds - min_elev) / (max_elev - min_elev)
    scale_x = src.res[0] * sample_factor
    scale_y = src.res[1] * sample_factor
    map_shape = arr_norm.shape
    xy_scale = (scale_x, scale_y)
    elev_range = (min_elev, max_elev)
    # Document DEM file size (MB)
    dem_size_mb = round(os.path.getsize(img_file) / (1024*1024), 2)

    # Extra validation: check for degenerate DEM
    if arr_ds.size == 0:
        print("ERROR: DEM sampled array (arr_ds) is EMPTY. Bad DEM file or incorrect sample_factor.")
    elif np.all(np.isnan(arr_ds)):
        print("ERROR: DEM sampled array is all NaN! No valid data after decimation.")
    elif min_elev == max_elev:
        print("WARNING: DEM has no elevation variation (flat tile)")

height_map = arr_norm
dem_file = img_file
crs = crs
profile = profile
nodata = nodata
min_elev = min_elev
max_elev = max_elev
mean_elev = mean_elev
n_nodata = n_nodata
map_shape = map_shape
xy_scale = xy_scale
sample_factor = sample_factor
dem_size_mb = dem_size_mb
dem_ready = True