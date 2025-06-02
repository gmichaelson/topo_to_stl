# Summarize rasterize-based mesh export
stl_file = rasterize_mesh_stats['stl_file']
info = f"""
Rasterize mask Nevada STL report
------------------------------------
- File: {stl_file}
- Vertex count: {rasterize_mesh_stats['num_verts']}
- Face count: {rasterize_mesh_stats['num_faces']}
- XY (mm): {rasterize_mesh_stats['size_xy'][0]:.2f} x {rasterize_mesh_stats['size_xy'][1]:.2f}
- Z (min): {rasterize_mesh_stats['min_z']:.2f} mm, Z (max): {rasterize_mesh_stats['max_z']:.2f} mm
- Flat base: yes, Z=1mm
- Mask area frac: {mask_frac:.3f}

Success: rasterize_stl_ready={rasterize_stl_ready}
"""
rasterize_stl_audit_summary = info
rasterize_stl_report_ok = rasterize_stl_ready and (rasterize_mesh_stats['num_faces'] > 0)
rasterize_stl_export_path = stl_file if rasterize_stl_report_ok else None
