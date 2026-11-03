"""
================================================================================
09_GEE_Basin_Rainfall.py
Workflow Step 09: Basin-Wide Annual CHIRPS Rainfall Extraction (1990-2026)
Manuscript Section: 2.4 Impoundment construction trends

Description:
  Extracts the 37-year annual rainfall time series (1990-2026) from the
  Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS)
  across the Ugab River basin in Google Earth Engine.

  Contextualises the historical tempo of impoundment establishment against
  catchment-wide hydrometeorological droughts and pluvial periods (e.g., 1992,
  2010, and recent extremes).

Inputs:
  - 'Data/Ugab_Cleaned_Dams_Final.shp' (from Step 07)
  - 'Data/Ugab_Dams_Landsat_Cleaned_TimeSeries.csv' (from Step 08)

Outputs:
  - 'Data/Ugab_Annual_Basin_Rainfall_1990_2026.csv'
  - 'Data/Rainfall_vs_Construction_Timeline.png'
================================================================================
"""

import os
import json
import ee
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt

# ============================================================
# 0. CONFIGURATION & INITIALISATION
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

# Replace 'YOUR_GEE_PROJECT' with your Google Earth Engine project ID
GEE_PROJECT = 'YOUR_GEE_PROJECT'

try:
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine initialised successfully.")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine authenticated and initialised successfully.")

# Input file paths
shp_path = os.path.join(DATA_DIR, 'Ugab_Cleaned_Dams_Final.shp')
const_path = os.path.join(DATA_DIR, 'Ugab_Dams_Landsat_Cleaned_TimeSeries.csv')

out_csv = os.path.join(DATA_DIR, 'Ugab_Annual_Basin_Rainfall_1990_2026.csv')
out_plot = os.path.join(DATA_DIR, 'Rainfall_vs_Construction_Timeline.png')

# ============================================================
# 1. DEFINE BASIN GEOMETRY
# ============================================================
print(f"Loading dam locations from {shp_path} to construct basin bounding geometry...")
gdf = gpd.read_file(shp_path)
basin_geom = shapely.force_2d(gdf.geometry.unary_union.convex_hull)
geojson = json.loads(gpd.GeoSeries([basin_geom]).to_json())['features'][0]['geometry']
ee_geom = ee.Geometry(geojson)

# ============================================================
# 2. CHIRPS ANNUAL RAINFALL EXTRACTION
# ============================================================
print("Extracting annual CHIRPS rainfall (1990-2026)...")
try:
    chirps = ee.ImageCollection('UCSB-CHC/CHIRPS/V3/DAILY_RNL')
    band_name = chirps.first().bandNames().get(0).getInfo()
    print("Using CHIRPS V3 DAILY_RNL dataset.")
except Exception:
    chirps = ee.ImageCollection('UCSB-CHC/CHIRPS/DAILY')
    band_name = 'precipitation'
    print("Using CHIRPS V2.0 DAILY dataset.")

years = range(1990, 2027)
results = []

for year in years:
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    yearly_sum = chirps.filterDate(start_date, end_date).select(band_name).sum()
    mean_rainfall = yearly_sum.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee_geom,
        scale=5000,
        maxPixels=1e9
    ).getInfo()
    
    val = mean_rainfall.get(band_name)
    results.append({'Year': year, 'Rainfall_mm': val})
    if val is not None:
        print(f"  {year}: {val:.1f} mm")

rain_df = pd.DataFrame(results)
rain_df.to_csv(out_csv, index=False)
print(f"Annual basin rainfall data saved to: {out_csv}")

# ============================================================
# 3. CONTEXTUAL COMPARISON FIGURE
# ============================================================
if os.path.exists(const_path):
    print("Generating contextual rainfall vs. construction timeline chart...")
    df_l8 = pd.read_csv(const_path)
    mndwi_col = [c for c in df_l8.columns if 'MNDWI_max' in c or 'MNDWI' in c][0]
    
    # Establish construction years using the conservative remote sensing threshold (MNDWI > 0.05)
    built_years = []
    for dam_id, group in df_l8.groupby('id'):
        wy = group[group[mndwi_col] > 0.05].sort_values('Year')['Year'].values
        if len(wy) > 0:
            built_years.append(int(wy[0]))
            
    counts = pd.Series(built_years).value_counts().sort_index()
    counts = counts.reindex(range(1990, 2027), fill_value=0)
    
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    
    # Annual Rainfall
    ax1.plot(rain_df['Year'], rain_df['Rainfall_mm'], color='#1f78b4', marker='o', linewidth=2, label='Annual Rainfall (CHIRPS)')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Annual Rainfall (mm)', color='#1f78b4', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#1f78b4')
    
    # Drought markers
    ax1.axvline(x=1992, color='#e31a1c', linestyle='--', alpha=0.6, label='Major Droughts (1992, 2013, 2019)')
    ax1.axvline(x=2013, color='#e31a1c', linestyle='--', alpha=0.6)
    ax1.axvline(x=2019, color='#e31a1c', linestyle='--', alpha=0.6)
    
    # Newly detectable dams (excluding the <=1990 baseline)
    ax2 = ax1.twinx()
    counts_sub = counts[counts.index > 1990]
    ax2.bar(counts_sub.index, counts_sub.values, width=0.6, color='#ff7f00', alpha=0.65, label='New Impoundments Detected')
    ax2.set_ylabel('New Impoundments Established', color='#ff7f00', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#ff7f00')
    
    plt.title('Ugab Basin: Annual Rainfall and Remote Impoundment Establishment (1991-2026)', pad=12, fontsize=12)
    fig.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Contextual comparison plot saved to: {out_plot}")
