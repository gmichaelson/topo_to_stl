import geopandas as gpd
import os
import urllib.request

nevada_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip"
shp_zip_path = "cb_2022_us_state_500k.zip"
shp_dir = "cb_2022_us_state_500k/"

def download_and_extract(url, out_zip, out_folder):
    if not os.path.exists(out_zip):
        urllib.request.urlretrieve(url, out_zip)
    if not os.path.exists(out_folder):
        import zipfile
        with zipfile.ZipFile(out_zip, 'r') as zip_ref:
            zip_ref.extractall(out_folder)

download_and_extract(nevada_url, shp_zip_path, shp_dir)

# Load shapefile
shp_file = os.path.join(shp_dir, "cb_2022_us_state_500k.shp")
gdf = gpd.read_file(shp_file)

# Filter for Nevada (STATEFP = '32')
nevada = gdf[gdf['STATEFP'] == '32'].to_crs("EPSG:4326")

# Optionally further clip to DEM bbox in next block
nevada_boundary = nevada.geometry.iloc[0]

gdf_nevada = nevada
nevada_loaded = True
