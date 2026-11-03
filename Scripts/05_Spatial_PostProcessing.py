"""
================================================================================
05_Spatial_PostProcessing.py
Workflow Step 05: Vectorization, Spatial Masking & Validation in Ugab Basin
Manuscript Section: 2.3.3 Candidate pixel post-processing & 2.3.4 Validation in the Ugab basin

Description:
  Takes the classified candidate raster from Step 04 and performs spatial
  cleaning, clustering, and exclusion masking:
    1. Vectorizes contiguous classified pixels into polygons.
    2. Clusters and merges nearby polygons within 100 meters (buffer +50m, union, buffer -50m).
    3. Excludes false positives using authoritative spatial masks in Data/:
       - OpenStreetMap buildings, urban land use, and roads
       - Global mining polygons (100m buffer)
       - HydroLAKES natural water bodies (100m buffer)
       - Seasonal standing pans and vleis (50-100m buffer)
       - High-flow MERIT Hydro river channels (50m buffer)
    4. Evaluates slope (<= 8 deg) and Height Above Nearest Drainage (<= 12 m).
    5. Validates candidate detections against the independent Atlas of Namibia
       inventory in the Ugab Basin (n = 641 active dams).

Outputs:
  - Vector shapefile: 'Data/Ugab_Dam_Candidates_Vector.shp'
  - Note: This shapefile must be uploaded to Google Earth Engine Assets for Step 06.
================================================================================
"""

import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import numpy as np
import ee

# ============================================================
# 1. FILE CONFIGURATIONS
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

# Replace 'YOUR_GEE_PROJECT' with your Google Earth Engine project ID
GEE_PROJECT = 'YOUR_GEE_PROJECT'

# Input Raster and Boundaries
tif_path = os.path.join(DATA_DIR, 'Ugab_Dam_Candidates_Raw.tif')
basin_path = os.path.join(DATA_DIR, 'Ugab_Basin_Boundary.gpkg')
atlas_path = os.path.join(DATA_DIR, 'Ugab_Reference_Dams.gpkg')

# Exclusion Mask Datasets
mining_path = os.path.join(DATA_DIR, 'Ugab_Mask_Mining.gpkg')
lakes_path = os.path.join(DATA_DIR, 'Ugab_Mask_Lakes.gpkg')
pans_path = os.path.join(DATA_DIR, 'Ugab_Mask_Large_Pans.gpkg')
small_pans_path = os.path.join(DATA_DIR, 'Ugab_Mask_Small_Pans.gpkg')
urban_path = os.path.join(DATA_DIR, 'Ugab_Mask_Urban.gpkg')
merit_path = os.path.join(DATA_DIR, 'Ugab_Mask_Major_Rivers.gpkg')
osm_buildings_path = os.path.join(DATA_DIR, 'Ugab_Mask_OSM_Buildings.gpkg')
osm_urban_path = os.path.join(DATA_DIR, 'Ugab_Mask_OSM_Urban.gpkg')
osm_roads_path = os.path.join(DATA_DIR, 'Ugab_Mask_OSM_Roads.gpkg')

out_shp = os.path.join(DATA_DIR, 'Ugab_Dam_Candidates_Vector.shp')

# ============================================================
# 2. LOAD SUBBASIN AND ANCILLARY MASKS
# ============================================================
print("Loading Subbasin & Ancillary Exclusion Masks from Data folder...")
basin_gdf = gpd.read_file(basin_path).to_crs("EPSG:4326")

mining_gdf = gpd.read_file(mining_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(mining_path) else gpd.GeoDataFrame()
if not mining_gdf.empty: mining_gdf.geometry = mining_gdf.geometry.buffer(100)

lakes_gdf = gpd.read_file(lakes_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(lakes_path) else gpd.GeoDataFrame()
if not lakes_gdf.empty: lakes_gdf.geometry = lakes_gdf.geometry.buffer(100)

pans_gdf = gpd.read_file(pans_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(pans_path) else gpd.GeoDataFrame()
if not pans_gdf.empty: pans_gdf.geometry = pans_gdf.geometry.buffer(100)

small_pans_gdf = gpd.read_file(small_pans_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(small_pans_path) else gpd.GeoDataFrame()
if not small_pans_gdf.empty: small_pans_gdf.geometry = small_pans_gdf.geometry.buffer(50)

urban_gdf = gpd.read_file(urban_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(urban_path) else gpd.GeoDataFrame()
if not urban_gdf.empty: urban_gdf.geometry = urban_gdf.geometry.buffer(100)

merit_gdf = gpd.read_file(merit_path).to_crs("EPSG:3857") if os.path.exists(merit_path) else gpd.GeoDataFrame()
if not merit_gdf.empty: merit_gdf.geometry = merit_gdf.geometry.buffer(50)

osm_b_gdf = gpd.read_file(osm_buildings_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(osm_buildings_path) else gpd.GeoDataFrame()
osm_u_gdf = gpd.read_file(osm_urban_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(osm_urban_path) else gpd.GeoDataFrame()
osm_r_gdf = gpd.read_file(osm_roads_path, mask=basin_gdf).to_crs("EPSG:3857") if os.path.exists(osm_roads_path) else gpd.GeoDataFrame()
if not osm_r_gdf.empty: osm_r_gdf.geometry = osm_r_gdf.geometry.buffer(20)

# ============================================================
# 3. VECTORIZE RASTER CANDIDATES
# ============================================================
print(f"Vectorizing raster candidates from: {tif_path}")
with rasterio.open(tif_path) as src:
    image = src.read(1)
    mask = image > 0
    geoms = [{'properties': {'raster_val': v}, 'geometry': s}
             for s, v in shapes(image, mask=mask, transform=src.transform)]

gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
gdf = gdf[gdf.geometry.type == 'Polygon']
basin_gdf_proj = basin_gdf.to_crs(gdf.crs)
gdf_clipped = gpd.overlay(gdf, basin_gdf_proj, how='intersection')
gdf_proj = gdf_clipped.to_crs("EPSG:3857")

# ============================================================
# 4. SPATIAL CLUSTERING (100M MORPHOLOGICAL MERGING)
# ============================================================
print("Applying 100m spatial clustering and union...")
buffered = gdf_proj.geometry.buffer(50)
unioned = buffered.union_all()
merged = unioned.buffer(-50)
if merged.geom_type == 'MultiPolygon':
    merged_geoms = list(merged.geoms)
elif merged.geom_type == 'Polygon':
    merged_geoms = [merged]
else:
    merged_geoms = []

gdf_proj = gpd.GeoDataFrame(geometry=merged_geoms, crs="EPSG:3857")
print(f"Candidate polygon clusters formed: {len(gdf_proj)}")

# ============================================================
# 5. APPLY EXCLUSION MASKS
# ============================================================
print("Filtering candidate polygons against exclusion masks...")
masks_to_apply = [mining_gdf, lakes_gdf, pans_gdf, small_pans_gdf, urban_gdf, merit_gdf, osm_b_gdf, osm_u_gdf, osm_r_gdf]
for mask_gdf in masks_to_apply:
    if not mask_gdf.empty:
        joined = gpd.sjoin(gdf_proj, mask_gdf, how='inner', predicate='intersects')
        gdf_proj = gdf_proj[~gdf_proj.index.isin(joined.index)]

print(f"Candidates remaining after mask exclusion: {len(gdf_proj)}")

# ============================================================
# 6. TOPOGRAPHY VALIDATION (GEE SAMPLING)
# ============================================================
print("Sampling slope and HAND for candidate centroids in Earth Engine...")
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

gdf_proj['id'] = gdf_proj.index
centroids = gdf_proj.to_crs("EPSG:4326").centroid
features = [
    ee.Feature(ee.Geometry.Point([geom.x, geom.y]), {'id': int(idx)})
    for idx, geom in centroids.items() if not geom.is_empty
]

fc = ee.FeatureCollection(features)
dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
slope = ee.Terrain.slope(dem).rename('slope')
hand = ee.Image("MERIT/Hydro/v1_0_1").select('hnd').rename('hand')
topo = slope.addBands(hand)
sampled = topo.sampleRegions(collection=fc, properties=['id'], scale=30, tileScale=16).getInfo()

rows = [{'id': f['properties'].get('id'), 'slope': f['properties'].get('slope'), 'hand': f['properties'].get('hand')}
        for f in sampled['features']]
df_topo = pd.DataFrame(rows)
merged_df = gdf_proj.merge(df_topo, on='id')

# Apply final Topography bounds (Slope <= 8 deg, HAND <= 12 m)
merged_df = merged_df[(merged_df['slope'] <= 8.0) & (merged_df['hand'] <= 12.0)].copy()
print(f"Candidates remaining after topography filter: {len(merged_df)}")

# ============================================================
# 7. VALIDATION AGAINST ATLAS OF NAMIBIA INVENTORY
# ============================================================
print("Calculating validation recall against Atlas of Namibia inventory...")
atlas = gpd.read_file(atlas_path).to_crs("EPSG:3857")
atlas_buffered = atlas.copy()
atlas_buffered.geometry = atlas.geometry.buffer(100)
joined = gpd.sjoin(merged_df, atlas_buffered, how='inner', predicate='intersects')

merged_df['is_true_dam'] = 0
merged_df.loc[joined.index, 'is_true_dam'] = 1

tp = len(merged_df[merged_df['is_true_dam'] == 1])
fp = len(merged_df[merged_df['is_true_dam'] == 0])
total = len(merged_df)
total_atlas = len(atlas)
recall = (tp / total_atlas) * 100
precision = (tp / total) * 100 if total > 0 else 0

print(f"\n--- UGAB BASIN VALIDATION METRICS ---")
print(f"Total Detected Candidates: {total}")
print(f"True Positives (Matching Atlas): {tp}")
print(f"Candidate Novel Dams / Unmapped: {fp}")
print(f"Recall against Active Atlas Dams: {tp} / {total_atlas} ({recall:.1f}%)")
print(f"Atlas Apparent Precision: {precision:.1f}%")

merged_df.to_crs("EPSG:4326").to_file(out_shp)
print(f"\nVector shapefile exported to: {out_shp}")
print(f"IMPORTANT: Please upload 'Ugab_Dam_Candidates_Vector.shp' as a GEE Asset ('projects/YOUR_GEE_PROJECT/assets/Ugab_Dam_Candidates_Vector') to run Step 06.")
