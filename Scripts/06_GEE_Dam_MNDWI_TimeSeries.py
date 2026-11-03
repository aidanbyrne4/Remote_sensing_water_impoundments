"""
================================================================================
06_GEE_Dam_MNDWI_TimeSeries.py
Workflow Step 06: Annual Maximum MNDWI Time-Series Extraction in Earth Engine
Manuscript Section: 2.3.3 Candidate pixel post-processing & 2.4 Impoundment construction trends

Description:
  Extracts multi-decadal annual maximum Modified Normalized Difference Water
  Index (MNDWI) time series across all candidate dam polygons in the Ugab Basin:
    1. Sentinel-2 Surface Reflectance (2016-2026, 10m resolution) using Cloud Score Plus.
    2. Landsat 5 TM, 7 ETM+, 8 OLI, and 9 OLI-2 (1990-2026, 30m resolution) Collection 2.

  Observations are restricted to the primary wet season (January to May) to
  capture episodic peak surface water extent and dam filling events.

Inputs:
  - GEE Asset: Uploaded candidate vector polygons from Step 05
    ('projects/YOUR_GEE_PROJECT/assets/Ugab_Dam_Candidates_Vector')

Outputs:
  - Exports to Google Drive:
    - 'Ugab_Dams_S2_TimeSeries_2016_2026.csv'
    - 'Ugab_Dams_Landsat_TimeSeries_1990_2026.csv'
================================================================================
"""

import ee

# ============================================================
# 0. INITIALIZE EARTH ENGINE
# ============================================================
# Replace 'YOUR_GEE_PROJECT' with your Google Earth Engine project ID
GEE_PROJECT = 'YOUR_GEE_PROJECT'

try:
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine initialised successfully.")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine authenticated and initialised successfully.")

# ============================================================
# 1. CANDIDATE VECTOR ASSET & PARAMETERS
# ============================================================
ASSET_ID = f"projects/{GEE_PROJECT}/assets/Ugab_Dam_Candidates_Vector"
dams_fc = ee.FeatureCollection(ASSET_ID)

# Peak wet-season window (January to May) to capture maximum surface inundation
START_MONTH = 1
END_MONTH = 5

# ============================================================
# 2. SENTINEL-2 ANNUAL MAXIMUM EXTRACTION (2016-2026)
# ============================================================
print("Processing Sentinel-2 optical time series (2016-2026)...")

def get_s2_mndwi(img):
    cs_cdf = img.select('cs_cdf')
    mask = cs_cdf.gte(0.60)
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    return mndwi.updateMask(mask).set('system:time_start', img.get('system:time_start'))

s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(dams_fc) \
    .filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month'))
cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
s2_linked = s2_col.linkCollection(cs_plus, ['cs_cdf'])
s2_mndwi_col = s2_linked.map(get_s2_mndwi)

def extract_s2_year(year):
    annual_col = s2_mndwi_col.filter(ee.Filter.calendarRange(year, year, 'year'))
    annual_img = annual_col.max().rename('MNDWI_max')
    annual_obs = annual_col.count().rename('valid_obs')
    return ee.Image.cat([annual_img, annual_obs]).set('Year', year)

s2_years = ee.List.sequence(2016, 2026)
s2_annual_images = ee.ImageCollection.fromImages(s2_years.map(extract_s2_year))

# ============================================================
# 3. LANDSAT ANNUAL MAXIMUM EXTRACTION (1990-2026)
# ============================================================
print("Processing Landsat optical time series (1990-2026)...")

def mask_landsat(img):
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return img.updateMask(cloud.And(shadow))

def get_l57_mndwi(img):
    img = mask_landsat(img)
    mndwi = img.normalizedDifference(['SR_B2', 'SR_B5']).rename('MNDWI')
    return mndwi.set('system:time_start', img.get('system:time_start'))

def get_l89_mndwi(img):
    img = mask_landsat(img)
    mndwi = img.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    return mndwi.set('system:time_start', img.get('system:time_start'))

l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(dams_fc).filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month')).map(get_l57_mndwi)
l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(dams_fc).filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month')).map(get_l57_mndwi)
l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(dams_fc).filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month')).map(get_l89_mndwi)
l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(dams_fc).filter(ee.Filter.calendarRange(START_MONTH, END_MONTH, 'month')).map(get_l89_mndwi)

landsat_mndwi_col = l5.merge(l7).merge(l8).merge(l9)

def extract_landsat_year(year):
    annual_col = landsat_mndwi_col.filter(ee.Filter.calendarRange(year, year, 'year'))
    annual_img = annual_col.max().rename('MNDWI_max')
    annual_obs = annual_col.count().rename('valid_obs')
    return ee.Image.cat([annual_img, annual_obs]).set('Year', year)

landsat_years = ee.List.sequence(1990, 2026)
landsat_annual_images = ee.ImageCollection.fromImages(landsat_years.map(extract_landsat_year))

# ============================================================
# 4. SPATIAL REDUCTION OVER POLYGONS
# ============================================================
print("Setting up spatial polygon reducers...")
combined_reducer = ee.Reducer.max().combine(
    reducer2=ee.Reducer.mean(), sharedInputs=True
).combine(
    reducer2=ee.Reducer.median(), sharedInputs=True
)

def reduce_regions(img, fc, scale):
    reduced = img.reduceRegions(
        collection=fc,
        reducer=combined_reducer,
        scale=scale,
        tileScale=4
    )
    year = img.get('Year')
    return reduced.map(lambda f: f.set('Year', year))

s2_extracted = s2_annual_images.map(lambda img: reduce_regions(img, dams_fc, 10)).flatten()
landsat_extracted = landsat_annual_images.map(lambda img: reduce_regions(img, dams_fc, 30)).flatten()

# ============================================================
# 5. SUBMIT EXPORT TASKS
# ============================================================
task_s2 = ee.batch.Export.table.toDrive(
    collection=s2_extracted,
    description='Ugab_Dams_S2_TimeSeries_2016_2026',
    fileNamePrefix='Ugab_Dams_S2_TimeSeries_2016_2026',
    folder='EarthEngineExports',
    fileFormat='CSV'
)
task_s2.start()
print(f"Sentinel-2 Export Task Started: {task_s2.id}")

task_landsat = ee.batch.Export.table.toDrive(
    collection=landsat_extracted,
    description='Ugab_Dams_Landsat_TimeSeries_1990_2026',
    fileNamePrefix='Ugab_Dams_Landsat_TimeSeries_1990_2026',
    folder='EarthEngineExports',
    fileFormat='CSV'
)
task_landsat.start()
print(f"Landsat Export Task Started: {task_landsat.id}")
print("\nMonitor tasks in the GEE Tasks tab. Download both CSVs into your local 'Data' folder.")
