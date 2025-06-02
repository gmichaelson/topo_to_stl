import numpy as np
import os
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from stl import mesh
import geopandas as gpd

# Inputs:
# arr_ds, profile, min_elev, max_elev, sample_factor, nevada_boundary, map_shape, scale_x, scale_y
H, W = map_shape

# 1. Project Nevada polygon to raster CRS (same as arr_ds/profile)
dem_crs = profile['crs']
if hasattr(nevada_boundary, 'crs') and nevada_boundary.crs != dem_crs:
    # Safe fallback if boundary has .crs
    poly_geom = nevada_boundary.to_crs(dem_crs)
else:
    # Manual conversion using gdf_nevada if needed
    gdf_boundary = gpd.GeoDataFrame(geometry=[nevada_boundary], crs='epsg:4326')
    poly_geom = gdf_boundary.to_crs(dem_crs).geometry.iloc[0]

# 2. geometry_mask (outputs True for OUTSIDE, False for INSIDE)
dem_transform = profile['transform']
mask_geom = geometry_mask([mapping(poly_geom)], out_shape=(H, W), transform=dem_transform, invert=True)

# 3. Mask DEM - values INSIDE Nevada, set OUTSIDE to NaN
masked_arr = np.array(arr_ds)
masked_arr[~mask_geom] = np.nan  # OUTSIDE is False so invert mask

# 4. Build mesh (reusing scaling logic from original Nevada final mesh):
x_scale, y_scale = scale_x, scale_y
xy_scale_factor = 50.8 / (np.ptp(np.arange(H) * scale_y) if np.ptp(np.arange(H) * scale_y) > np.ptp(np.arange(W) * scale_x) else np.ptp(np.arange(W) * scale_x)) / 1000
# Use minimum/maximum values on valid data for Z
valid_h = masked_arr[mask_geom]
min_h, max_h = np.nanmin(valid_h), np.nanmax(valid_h)
def z_affine_geom(h):
    # Map min->1mm, max->4mm
    if max_h == min_h:
        return np.ones_like(h) * 2.5
    return 1.0 + 3.0 * (h - min_h) / (max_h - min_h)

# Build top and base vertices
grid_x_m = np.arange(W) * x_scale + dem_transform[2]
grid_y_m = np.arange(H) * y_scale + dem_transform[5]
xx, yy = np.meshgrid(grid_x_m, grid_y_m)
verts_top = []
verts_base = []
vert_idx_map = -np.ones((H, W), dtype=int)
vert_count = 0
for i in range(H):
    for j in range(W):
        if mask_geom[i, j]:
            x_mm = (xx[i, j] - np.min(grid_x_m)) * xy_scale_factor * 1000
            y_mm = (yy[i, j] - np.min(grid_y_m)) * xy_scale_factor * 1000
            z_mm = z_affine_geom(np.array([masked_arr[i, j]]))[0]
            verts_top.append((x_mm, y_mm, z_mm))
            verts_base.append((x_mm, y_mm, 1.0))
            vert_idx_map[i, j] = vert_count
            vert_count += 1
verts_top = np.array(verts_top)
verts_base = np.array(verts_base)
verts = np.vstack([verts_top, verts_base])

# Only connect triangles where all three are inside
faces = []
for i in range(H - 1):
    for j in range(W - 1):
        ids = [vert_idx_map[i, j], vert_idx_map[i + 1, j], vert_idx_map[i, j + 1], vert_idx_map[i + 1, j + 1]]
        if ids[0] >= 0 and ids[1] >= 0 and ids[2] >= 0:
            faces.append([ids[0], ids[1], ids[2]])
        if ids[3] >= 0 and ids[2] >= 0 and ids[1] >= 0:
            faces.append([ids[3], ids[2], ids[1]])

# Skirt around outer perimeter (neighbor pixel outside mask)
skirt_faces = []
perim_pts = []
base_start = len(verts_top)
for i in range(H):
    for j in range(W):
        if mask_geom[i,j] and any(
            not mask_geom[ni,nj] if (0<=ni<H and 0<=nj<W) else True
            for ni,nj in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
        ):
            perim_pts.append((i,j))
            idx_top = vert_idx_map[i,j]
            idx_base = base_start + idx_top
            for di,dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                ii,jj = i+di,j+dj
                if 0<=ii<H and 0<=jj<W and mask_geom[ii,jj]:
                    idx_top2 = vert_idx_map[ii,jj]
                    idx_base2 = base_start + idx_top2
                    skirt_faces.append([idx_top, idx_top2, idx_base])
                    skirt_faces.append([idx_top2, idx_base2, idx_base])
faces = np.array(faces + skirt_faces)

# Export STL with name as requested
stl_name = "nevada_mask_geomask.stl"
gmask_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        gmask_mesh.vectors[i][j] = verts[f[j], :]
gmask_mesh.save(stl_name)

# Diagnostics
mesh_stats_geomask = {
    'num_verts': int(verts.shape[0]),
    'num_faces': int(faces.shape[0]),
    'stl_file': stl_name,
    'min_z': float(np.min(verts_top[:,2])) if verts_top.shape[0] else None,
    'max_z': float(np.max(verts_top[:,2])) if verts_top.shape[0] else None,
    'size_xy': [float(np.ptp(verts_top[:,0])), float(np.ptp(verts_top[:,1]))]
}
mesh_geomask_ready = os.path.exists(stl_name)
perim_count_geomask = len(perim_pts)
