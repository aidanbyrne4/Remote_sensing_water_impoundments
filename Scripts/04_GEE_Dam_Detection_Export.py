"""
================================================================================
04_GEE_Dam_Detection_Export.py
Workflow Step 04: Basin-Wide Dam Classification & Raster Export in Earth Engine
Manuscript Section: 2.3.2 Model development & 2.3.3 Candidate pixel post-processing

Description:
  Translates the 3-pathway decision tree rules (from Step 02) and topographical
  thresholds (from Step 03) into raster algebra in Google Earth Engine across
  the entire Ugab River basin.

  Decision Pathways Applied:
    Condition 1 (High optical): MNDWI_wet_freq > 0.07 & 3.48 < VVVH_wet_median <= 9.85
    Condition 2 (Weak optical): 0.01 < MNDWI_wet_freq <= 0.07 & VVVH_wet_median <= 8.21
    Condition 3 (Dry-season persistence): MNDWI_wet_freq <= 0.01 & MNDWI_dry_freq > 0.01

  Raster Filters Applied:
    - Height Above Nearest Drainage (MERIT HAND) < 12 m
    - Terrain Slope (NASADEM) < 8 degrees
    - Connected pixel threshold >= 4 contiguous 10-m pixels (>= 400 m²)

Output:
  - Exports GeoTIFF to Google Drive: 'Ugab_Dam_Candidates_Raw.tif'
================================================================================
"""

import os
import ee
import geopandas as gpd
import shapely

# ============================================================
# 0. CONFIGURATION & INITIALISATION
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
basin_path = os.path.join(DATA_DIR, 'Ugab_Basin_Boundary.gpkg')

# Replace 'YOUR_GEE_PROJECT' with your Google Earth Engine project ID
GEE_PROJECT = 'YOUR_GEE_PROJECT'

try:
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine initialised successfully.")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine authenticated and initialised successfully.")

# Load Ugab Basin Region of Interest
basin_gdf = gpd.read_file(basin_path).to_crs("EPSG:4326")
basin_gdf.geometry = shapely.force_2d(basin_gdf.geometry)
subbasin = ee.FeatureCollection(
    ee.Feature(ee.Geometry(basin_gdf.geometry.iloc[0].__geo_interface__))
).geometry()

# ============================================================
# 1. OPTICAL SENTINEL-2 COMPOSITES (WET & DRY SEASONS)
# ============================================================
print("Building Sentinel-2 optical composites...")
cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

def get_mndwi(img):
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    return img.addBands(mndwi)

# Wet Season (Jan - Mar, 2016-2025)
s2_wet = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(subbasin) \
    .filterDate('2016-01-01', '2025-01-01') \
    .filter(ee.Filter.calendarRange(1, 3, 'month')) \
    .select(['B3', 'B11'])

s2_wet_masked = s2_wet.linkCollection(cs_plus, ['cs_cdf']).map(
    lambda img: img.updateMask(img.select('cs_cdf').gte(0.60))
)
mndwi_wet_freq = s2_wet_masked.map(get_mndwi).map(
    lambda img: img.select('MNDWI').gt(0.1)
).mean().rename('MNDWI_wet_freq')

# Dry Season (May - Jul, 2016-2025)
s2_dry = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(subbasin) \
    .filterDate('2016-01-01', '2025-01-01') \
    .filter(ee.Filter.calendarRange(5, 7, 'month')) \
    .select(['B3', 'B11'])

s2_dry_masked = s2_dry.linkCollection(cs_plus, ['cs_cdf']).map(
    lambda img: img.updateMask(img.select('cs_cdf').gte(0.60))
)
mndwi_dry_freq = s2_dry_masked.map(get_mndwi).map(
    lambda img: img.select('MNDWI').gt(0.1)
).mean().rename('MNDWI_dry_freq')

# ============================================================
# 2. SENTINEL-1 SAR COMPOSITES (WET SEASON)
# ============================================================
print("Building Sentinel-1 SAR composites...")
s1_wet = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filterBounds(subbasin) \
    .filterDate('2016-01-01', '2025-01-01') \
    .filter(ee.Filter.calendarRange(1, 3, 'month')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW'))

def calc_ratio(img):
    vv_minus_vh = img.select('VV').subtract(img.select('VH')).rename('VVVH')
    return img.addBands(vv_minus_vh)

vvvh_wet_median = s1_wet.map(calc_ratio).select('VVVH').median().rename('VVVH_wet_median')

# ============================================================
# 3. TOPOGRAPHY & EXCLUSION MASKS
# ============================================================
merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
hnd = merit_hydro.select('hnd')

dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
slope = ee.Terrain.slope(dem)

worldcover = ee.ImageCollection("ESA/WorldCover/v200").mosaic()
valid_classes = worldcover.neq(50)  # Exclude urban built areas

# ============================================================
# 4. APPLY COMBINATORIAL DECISION TREE RULES
# ============================================================
print("Applying decision tree raster rules...")
candidates = ee.Image(1).byte()

# Condition 1: High optical water signal (relaxed radar bounds)
cond_1 = mndwi_wet_freq.gt(0.07).And(vvvh_wet_median.lte(9.85)).And(vvvh_wet_median.gt(3.48))

# Condition 2: Weak optical water signal (strict radar threshold)
cond_2 = mndwi_wet_freq.lte(0.07).And(mndwi_wet_freq.gt(0.01)).And(vvvh_wet_median.lte(8.21))

# Condition 3: Dry season persistence (no radar constraint required)
cond_3 = mndwi_wet_freq.lte(0.01).And(mndwi_dry_freq.gt(0.01))

candidates = candidates.updateMask(cond_1.Or(cond_2).Or(cond_3))

# Topography Pre-filters
candidates = candidates.updateMask(hnd.lt(12))
candidates = candidates.updateMask(slope.lt(8))
candidates = candidates.updateMask(valid_classes)

# Spatial Noise Filter: At least 4 connected 10-m pixels (400 m²)
pixel_count = candidates.connectedPixelCount(100, False)
final_candidates = candidates.updateMask(pixel_count.gte(4)).rename('dam_candidate').clip(subbasin)

# ============================================================
# 5. EXPORT RASTER TO GOOGLE DRIVE
# ============================================================
export_task = ee.batch.Export.image.toDrive(
    image=final_candidates,
    description='Ugab_Dam_Candidates_Raw_Export',
    folder='EarthEngineExports',
    fileNamePrefix='Ugab_Dam_Candidates_Raw',
    region=subbasin,
    scale=10,
    maxPixels=1e13,
    fileFormat='GeoTIFF'
)

export_task.start()
print(f"Export task submitted! Task ID: {export_task.id}")
print("Monitor status in GEE Tasks. Download 'Ugab_Dam_Candidates_Raw.tif' to your 'Data' folder to proceed with Step 05.")
