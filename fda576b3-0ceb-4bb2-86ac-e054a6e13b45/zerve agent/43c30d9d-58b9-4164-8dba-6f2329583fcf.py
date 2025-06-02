import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Use arr_ds (sampled DEM for analysis)
# Use arr (full DEM if needed)
# Inputs from upstream: arr_ds, min_elev, max_elev, mean_elev, map_shape, dem_file

def safe_unique(a):
    # Only count unique for valid values
    return np.unique(a[~np.isnan(a)])

unique_elevs = safe_unique(arr_ds)
num_unique = len(unique_elevs)

# Quick preview - grayscale DEM visualization
fig, ax = plt.subplots(figsize=(5,5))
show = ax.imshow(arr_ds, cmap="gray", vmin=np.nanmin(arr_ds), vmax=np.nanmax(arr_ds))
plt.axis('off')
plt.tight_layout()
img_path = "dem_preview.png"
fig.savefig(img_path, bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Optional: Dynamic range check for hillshade effect (skip for speed/cost, grayscale is clear)

# Report core outputs
visual_path = img_path
stat_min = float(np.nanmin(arr_ds))
stat_max = float(np.nanmax(arr_ds))
stat_mean = float(np.nanmean(arr_ds))
stat_shape = arr_ds.shape
stat_unique_count = num_unique
file_name = dem_file
region_name = "Nevada Tile N39W117"

# For summary generation downstream
visual_stats_done = True