import numpy as np
from rasterio.features import rasterize
from shapely.geometry import mapping
import rasterio

# Inputs:
#  - arr_ds (decimated DEM 2D array)
#  - profile (from rasterio reader)
#  - sample_factor (int, down-sampling, from preprocess_dem)
#  - gdf_nevada (GeoDataFrame with Nevada projected)
#  - nevada_boundary (Geo series, original CRS)
#
# Output: mask_rasterize (bool 2D array same shape as arr_ds), arr_masked_rast (masked, values outside == min value)

# Get DEM shape and transform for sampled data
H, W = arr_ds.shape
sf = sample_factor
# For decimation, compute transform of sampled DEM grid
tran_full = profile['transform']
tran_sampled = rasterio.Affine(tran_full.a*sf, tran_full.b, tran_full.c, tran_full.d, tran_full.e*sf, tran_full.f)

# Project Nevada boundary to DEM CRS
crs_dem = profile['crs']
nevada_geom = gdf_nevada.to_crs(crs_dem)

# Rasterize Nevada polygon (value=1 inside, 0 outside)
geoms = [(mapping(shape), 1) for shape in nevada_geom.geometry]
mask_rasterize = rasterize(geoms, out_shape=(H, W), fill=0, transform=tran_sampled, dtype='uint8') == 1

# Masked DEM: keep DEM values inside, outside set to min inside
min_val = np.nanmin(arr_ds)
arr_masked_rast = np.where(mask_rasterize, arr_ds, min_val)
num_inside = int(mask_rasterize.sum())
mask_frac = float(num_inside) / (H*W)

rasterize_mask_ready = True
