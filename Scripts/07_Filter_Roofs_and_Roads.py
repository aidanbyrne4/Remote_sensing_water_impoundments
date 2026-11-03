"""
================================================================================
07_Filter_Roofs_and_Roads.py
Workflow Step 07: Temporal Filtering for Persistent Infrastructure (Roofs & Roads)
Manuscript Section: 2.3.3 Candidate pixel post-processing

Description:
  Uses the complete 2016-2026 Sentinel-2 record to distinguish true ephemeral
  water impoundments from smooth, highly reflective anthropogenic surfaces:
    1. Never Filled Filter: Maximum annual MNDWI <= 0.0 across all years.
    2. Static Variance Filter: Temporal standard deviation of MNDWI < 0.02
       (eliminating permanently stable reflective rooftops and asphalt).
    3. Road Filter: Minimal annual amplitude (< 0.25) and low peak (< 0.30).
    4. Roof Construction Step-Change Filter: Abrupt jump after 2016 that never
       dries out (post-construction minimum > -0.05, amplitude > 0.4, peak < 0.5).

Inputs:
  - 'Data/Ugab_Dams_S2_TimeSeries_2016_2026.csv' (from Step 06)
  - 'Data/Ugab_Dam_Candidates_Vector.shp' (from Step 05)

Outputs:
  - 'Data/Ugab_Cleaned_Dams_Final.shp' (758 verified active artificial impoundments)
================================================================================
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np

# ============================================================
# 1. FILE PATHS & DATA LOADING
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"

csv_path = os.path.join(DATA_DIR, 'Ugab_Dams_S2_TimeSeries_2016_2026.csv')
shp_path = os.path.join(DATA_DIR, 'Ugab_Dam_Candidates_Vector.shp')
out_shp = os.path.join(DATA_DIR, 'Ugab_Cleaned_Dams_Final.shp')

print(f"Loading Sentinel-2 time series from: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Loading candidate polygons from: {shp_path}")
gdf = gpd.read_file(shp_path)

id_col_csv = 'id' if 'id' in df.columns else 'system:index'
id_col_gdf = 'id' if 'id' in gdf.columns else 'FID'
mndwi_col = [c for c in df.columns if 'MNDWI_max' in c or 'MNDWI' in c][0]

# ============================================================
# 2. TEMPORAL DYNAMICS FILTERING
# ============================================================
print("Applying temporal variance and step-change infrastructure filters...")
results = []

for dam_id, group in df.groupby(id_col_csv):
    group = group.sort_values('Year')
    mndwi_vals = group[mndwi_col].dropna().values
    
    if len(mndwi_vals) == 0:
        continue
        
    amp = np.max(mndwi_vals) - np.min(mndwi_vals)
    max_val = np.max(mndwi_vals)
    mndwi_std = np.std(mndwi_vals)
    
    water_years = group[group[mndwi_col] > 0.0]['Year'].values
    first_water = water_years[0] if len(water_years) > 0 else -1
    
    post_min = -999.0
    if first_water > 2016:
        post_vals = group[group['Year'] >= first_water][mndwi_col].values
        if len(post_vals) > 0:
            post_min = np.min(post_vals)
        
    flag = "Keep"
    
    # 1. Never Filled Filter (Max MNDWI never exceeded 0.0)
    if max_val <= 0.0:
        flag = "Drop: Never Filled"
        
    # 2. Extreme Static Variance Filter (Reflective rooftops)
    elif mndwi_std < 0.02:
        flag = "Drop: Extreme Static Variance"
        
    # 3. Road Filter (Low fluctuating peak)
    elif amp < 0.25 and max_val < 0.30:
        flag = "Drop: Suspect Road"
        
    # 4. Roof Step-Change Filter (Permanent jump after 2016, never dries out)
    elif first_water > 2016 and post_min > -0.05 and amp > 0.4 and max_val < 0.5:
        flag = "Drop: Suspect Roof"
        
    results.append({
        id_col_csv: dam_id,
        'filter_flag': flag
    })

metrics_df = pd.DataFrame(results)

# Merge flags into GeoDataFrame
gdf = gdf.merge(metrics_df, left_on=id_col_gdf, right_on=id_col_csv, how='left')
missing_mask = gdf['filter_flag'].isna()
if missing_mask.sum() > 0:
    gdf.loc[missing_mask, 'filter_flag'] = "Keep"

flag_counts = gdf['filter_flag'].value_counts()
print("\n--- Infrastructure Filtering Summary ---")
for flag, count in flag_counts.items():
    print(f"  {flag}: {count} polygons")

cleaned_gdf = gdf[gdf['filter_flag'] == "Keep"].copy()
cleaned_gdf = cleaned_gdf.drop(columns=['filter_flag'])
if id_col_csv in cleaned_gdf.columns and id_col_csv != id_col_gdf:
    cleaned_gdf = cleaned_gdf.drop(columns=[id_col_csv])

print(f"\nSaving final cleaned dam dataset to: {out_shp}")
cleaned_gdf.to_file(out_shp)
print(f"Successfully saved {len(cleaned_gdf)} verified active artificial impoundments!")
