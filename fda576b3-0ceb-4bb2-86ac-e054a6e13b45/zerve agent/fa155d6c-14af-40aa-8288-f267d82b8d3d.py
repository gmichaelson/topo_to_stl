import numpy as np
from stl import mesh

# Inputs:
#  - arr_masked_rast (DEM after rasterize mask)
#  - mask_rasterize (bool mask)
#  - map_shape, scale_x, scale_y, sample_factor
#
H, W = arr_masked_rast.shape

# --- Check for empty/degenerate/invalid mask ---
if mask_rasterize is None or not isinstance(mask_rasterize, np.ndarray) or not np.any(mask_rasterize):
    print("[ERROR] Rasterized Nevada mask is empty! No points inside boundary.")
    rasterize_stl_ready = False
    rasterize_stl_path = None
    verts_top = np.zeros((0,3))
    faces_top = []
    skirt_faces = []
    rasterize_mesh_stats = {
        'num_verts': 0,
        'num_faces': 0,
        'min_z': None,
        'max_z': None,
        'size_xy': [0,0],
        'stl_file': None
    }
else:
    tran_full = profile['transform']
    sf = sample_factor
    tran_sampled = rasterio.Affine(tran_full.a*sf, tran_full.b, tran_full.c, tran_full.d, tran_full.e*sf, tran_full.f)
    x_coords = np.arange(W) * scale_x + tran_sampled.c
    y_coords = np.arange(H) * scale_y + tran_sampled.f
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    grid_idxs = np.argwhere(mask_rasterize)
    x_vals = xx[mask_rasterize]
    y_vals = yy[mask_rasterize]
    if x_vals.size == 0 or y_vals.size == 0:
        print("[ERROR] No grid points inside rasterized Nevada mask!")
        rasterize_stl_ready = False
        rasterize_stl_path = None
        verts_top = np.zeros((0,3))
        faces_top = []
        skirt_faces = []
        rasterize_mesh_stats = {
            'num_verts': 0,
            'num_faces': 0,
            'min_z': None,
            'max_z': None,
            'size_xy': [0,0],
            'stl_file': None
        }
    else:
        size_x_m = np.ptp(x_vals)
        size_y_m = np.ptp(y_vals)
        req_tall_mm = 50.8
        curr_tall_m = size_y_m if size_y_m >= size_x_m else size_x_m
        xy_scale_factor = req_tall_mm / (curr_tall_m * 1000) if curr_tall_m != 0 else 1.0

        valid_elev = arr_masked_rast[mask_rasterize]
        if valid_elev.size == 0 or np.all(np.isnan(valid_elev)):
            print("[ERROR] Rasterized DEM has no valid elevation values within mask!")
            rasterize_stl_ready = False
            rasterize_stl_path = None
            verts_top = np.zeros((0,3))
            faces_top = []
            skirt_faces = []
            rasterize_mesh_stats = {
                'num_verts': 0,
                'num_faces': 0,
                'min_z': None,
                'max_z': None,
                'size_xy': [0,0],
                'stl_file': None
            }
        else:
            min_h = float(np.min(valid_elev))
            max_h = float(np.max(valid_elev))
            def z_affine(h):
                if max_h == min_h:
                    return np.ones_like(h) * 2.5
                return 1.0 + 3.0 * (h - min_h)/(max_h - min_h)

            verts_top = []
            verts_base = []
            vert_idx_grid = -np.ones((H, W), dtype=int)
            for idx, (i, j) in enumerate(grid_idxs):
                x_m = xx[i, j]
                y_m = yy[i, j]
                x_mm = (x_m - np.min(x_vals)) * xy_scale_factor * 1000
                y_mm = (y_m - np.min(y_vals)) * xy_scale_factor * 1000
                z_mm = z_affine(np.array([arr_masked_rast[i, j]]))[0]  # [1,4]
                verts_top.append((x_mm, y_mm, z_mm))
                verts_base.append((x_mm, y_mm, 1.0))
                vert_idx_grid[i, j] = idx
            verts_top = np.array(verts_top)
            verts_base = np.array(verts_base)
            verts = np.vstack([verts_top, verts_base])

            faces_top = []
            for i, j in grid_idxs:
                if i+1 < H and j+1 < W:
                    idx0 = vert_idx_grid[i, j]
                    idx1 = vert_idx_grid[i+1, j]
                    idx2 = vert_idx_grid[i, j+1]
                    idx3 = vert_idx_grid[i+1, j+1]
                    if all(k >= 0 for k in [idx0, idx1, idx2]):
                        faces_top.append([idx0, idx1, idx2])
                    if all(k >= 0 for k in [idx3, idx2, idx1]):
                        faces_top.append([idx3, idx2, idx1])
            skirt_faces = []
            base_start = verts_top.shape[0]
            for idx, (i, j) in enumerate(grid_idxs):
                pad = [(-1,0),(1,0),(0,-1),(0,1)]
                if any(i+di < 0 or i+di >= H or j+dj < 0 or j+dj >= W or not mask_rasterize[i+di,j+dj] for di,dj in pad):
                    idx_top = vert_idx_grid[i,j]
                    idx_base = base_start + idx
                    for di,dj in pad:
                        ni, nj = i+di, j+dj
                        if 0<=ni<H and 0<=nj<W and mask_rasterize[ni, nj]:
                            idx_top2 = vert_idx_grid[ni, nj]
                            idx_base2 = base_start + np.where((grid_idxs == [ni, nj]).all(axis=1))[0][0]
                            skirt_faces.append([idx_top, idx_top2, idx_base])
                            skirt_faces.append([idx_top2, idx_base2, idx_base])
            faces = np.array(faces_top + skirt_faces) if verts_top.shape[0]>0 else np.zeros((0,3),dtype=int)
            if verts_top.shape[0]==0 or faces.shape[0]==0:
                print("[ERROR] No valid mesh could be generated from rasterized mask. Empty verts/faces.")
                rasterize_stl_ready = False
                rasterize_stl_path = None
                rasterize_mesh_stats = {
                    'num_verts': 0,
                    'num_faces': 0,
                    'min_z': None,
                    'max_z': None,
                    'size_xy': [0,0],
                    'stl_file': None
                }
            else:
                stl_filename = "nevada_mask_rasterize.stl"
                mesh_out = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        mesh_out.vectors[i][j] = verts[f[j], :]
                mesh_out.save(stl_filename)
                rasterize_stl_ready = True
                rasterize_stl_path = stl_filename
                rasterize_mesh_stats = {
                    'num_verts': verts.shape[0],
                    'num_faces': faces.shape[0],
                    'min_z': float(np.min(verts[:,2])),
                    'max_z': float(np.max(verts[:,2])),
                    'size_xy': [float(np.ptp(verts_top[:,0])), float(np.ptp(verts_top[:,1]))],
                    'stl_file': rasterize_stl_path
                }
