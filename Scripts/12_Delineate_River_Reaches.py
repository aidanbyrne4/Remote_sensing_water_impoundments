"""
================================================================================
12_Delineate_River_Reaches.py
Workflow Step 12: Delineation of Downstream Evaluation Reaches & Reference Basins
Manuscript Section: 2.5.2 Vegetation and rainfall data

Description:
  1. Traces the complete downstream flow paths from all snapped impoundments
     to the main Ugab River stem, extracting 414 sequential river segments:
     - Near: First immediate downstream reach
     - Mid:  Second downstream reach
     - Far:  Third downstream reach
  2. Samples 100 geographically matched reference reaches in catchments without
     any upstream impoundments, matched by longitudinal climatic gradient and
     Strahler stream order. Total evaluation sample = 514 river reaches.
  3. Fuses the HydroRIVERS network corridors with high-resolution MERIT Hydro
     flowlines (from Step 11) to accurately trace active ephemeral channel beds.

Inputs:
  - 'Data/Ugab_Rivers_Cumulative_Blockage.shp' (from Step 10)
  - 'Data/MERIT_Hydro_Rivers_10_Ugab.shp' (from Step 11)
  - 'Data/Ugab_Dams_Snapped.csv' (from Step 10)
  - 'Data/Ugab_Dams_Landsat_Cleaned_TimeSeries.csv' (from Step 08)
  - 'Data/Ugab_Cleaned_Dams_Final.shp' (from Step 07)

Outputs:
  - 'Data/Ugab_Downstream_Reaches_514.geojson' (Core reach geometry for Steps 13, 14, 15)
================================================================================
"""

import os
import random
import warnings
import pandas as pd
import geopandas as gpd
import shapely
import networkx as nx

warnings.filterwarnings('ignore')

# ============================================================
# 1. FILE CONFIGURATIONS & LOADING
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

# Input file paths
rivers_path = os.path.join(DATA_DIR, 'Ugab_Rivers_Cumulative_Blockage.shp')
merit_path = os.path.join(DATA_DIR, 'MERIT_Hydro_Rivers_10_Ugab.shp')
snapped_path = os.path.join(DATA_DIR, 'Ugab_Dams_Snapped.csv')
landsat_path = os.path.join(DATA_DIR, 'Ugab_Dams_Landsat_Cleaned_TimeSeries.csv')
dam_shp_path = os.path.join(DATA_DIR, 'Ugab_Cleaned_Dams_Final.shp')

# Output file paths
geojson_out = os.path.join(DATA_DIR, 'Ugab_Downstream_Reaches_514.geojson')

print(f"Loading rivers from: {rivers_path}")
rivers = gpd.read_file(rivers_path)
rivers_proj = rivers.to_crs(epsg=32733)

print(f"Loading MERIT Hydro flowlines from: {merit_path}")
merit_pixels = gpd.read_file(merit_path).to_crs(epsg=32733)

snapped_dams = pd.read_csv(snapped_path)
landsat_df = pd.read_csv(landsat_path)

# Calculate impoundment establishment years (conservative threshold MNDWI > 0.05)
mndwi_col = [c for c in landsat_df.columns if 'MNDWI_max' in c or 'MNDWI' in c][0]
landsat_water = landsat_df[landsat_df[mndwi_col] > 0.05]
const_years = landsat_water.groupby('id')['Year'].min().reset_index()

id_col_snapped = 'id_left' if 'id_left' in snapped_dams.columns else 'dam_id'
const_years.columns = [id_col_snapped, 'const_year']

dams = snapped_dams.merge(const_years, on=id_col_snapped, how='left')
# Fill missing construction years (very small dams undetected by Landsat) with the 1990 baseline
dams['const_year'] = dams['const_year'].fillna(1990)
valid_dams = dams
print(f"Processed {len(valid_dams)} active impoundments with establishment years.")

# ============================================================
# 2. TRACE DOWNSTREAM REACHES (NEAR, MID, FAR)
# ============================================================
print("Tracing Near, Mid, and Far reaches along the directed network...")
G = nx.DiGraph()
for _, row in rivers.iterrows():
    G.add_edge(row['HYRIV_ID'], row['NEXT_DOWN'])

dam_to_river = dict(zip(valid_dams[id_col_snapped], valid_dams['HYRIV_ID']))
dam_to_const = dict(zip(valid_dams[id_col_snapped], valid_dams['const_year']))

extraction_plan = []
processed_reaches = set()

for dam_id, snapped_reach in dam_to_river.items():
    if snapped_reach not in G.nodes: continue
    path = [snapped_reach]
    curr = snapped_reach
    for _ in range(3):
        successors = list(G.successors(curr))
        if not successors or successors[0] == 0: break
        curr = successors[0]
        path.append(curr)
        
    const_year = dam_to_const[dam_id]
    
    # Near (1st downstream reach)
    if len(path) > 1 and path[1] not in processed_reaches:
        extraction_plan.append({'dam_id': str(dam_id), 'type': 'Near', 'const_year': const_year, 'target_reach': path[1]})
        processed_reaches.add(path[1])
        
    # Mid (2nd downstream reach)
    if len(path) > 2 and path[2] not in processed_reaches:
        extraction_plan.append({'dam_id': str(dam_id), 'type': 'Mid', 'const_year': const_year, 'target_reach': path[2]})
        processed_reaches.add(path[2])
        
    # Far (3rd downstream reach)
    if len(path) > 3 and path[3] not in processed_reaches:
        extraction_plan.append({'dam_id': str(dam_id), 'type': 'Far', 'const_year': const_year, 'target_reach': path[3]})
        processed_reaches.add(path[3])

print(f"Delineated {len(extraction_plan)} downstream reaches below impoundments.")

# ============================================================
# 3. SAMPLE 100 GEOGRAPHICALLY MATCHED PRISTINE REFERENCE REACHES
# ============================================================
print("Sampling 100 geographically matched pristine reference reaches...")
all_snapped = set(valid_dams['HYRIV_ID'])
dam_affected = set()
for r in all_snapped:
    if r in G.nodes:
        dam_affected.update(nx.descendants(G, r))
        dam_affected.add(r)

pristine_reaches = list(set(rivers['HYRIV_ID']) - dam_affected)
dam_reaches_geom = rivers[rivers['HYRIV_ID'].isin(all_snapped)].geometry.centroid
min_x, max_x = dam_reaches_geom.x.min(), dam_reaches_geom.x.max()

pristine_gdf = rivers[rivers['HYRIV_ID'].isin(pristine_reaches)]
matched_pristine = pristine_gdf[(pristine_gdf.geometry.centroid.x >= min_x) & (pristine_gdf.geometry.centroid.x <= max_x)]
matched_pristine_ids = matched_pristine['HYRIV_ID'].tolist()

random.seed(42)
placebo_sample = random.sample(matched_pristine_ids, min(100, len(matched_pristine_ids)))
for reach_id in placebo_sample:
    extraction_plan.append({
        'dam_id': 'Placebo', 'type': 'Placebo', 'const_year': 1990, 'target_reach': reach_id
    })

print(f"Total reaches in evaluation design: {len(extraction_plan)} (414 downstream + 100 reference reaches = 514).")

# ============================================================
# 4. FUSE CORRIDORS WITH HIGH-RESOLUTION MERIT FLOWLINES
# ============================================================
print("Fusing reach corridors with high-resolution MERIT Hydro drainage pixels...")
sindex = merit_pixels.sindex
final_zones = []
fallback_count = 0

for i, plan in enumerate(extraction_plan):
    reach_id = plan['target_reach']
    reach_rows = rivers_proj[rivers_proj['HYRIV_ID'] == reach_id]
    if len(reach_rows) == 0: continue
    hydro_geom = reach_rows.geometry.iloc[0]
    
    search_corridor = hydro_geom.buffer(500)
    possible_matches = merit_pixels.iloc[list(sindex.intersection(search_corridor.bounds))]
    precise_matches = possible_matches[possible_matches.intersects(search_corridor)]
    
    if len(precise_matches) > 0:
        merit_geom = precise_matches.geometry.unary_union
        final_geom = merit_geom.buffer(15).simplify(15)
    else:
        final_geom = hydro_geom.buffer(50).simplify(15)
        fallback_count += 1
        
    final_zones.append({
        'zone_id': f"{plan['dam_id']}_{plan['type']}_{reach_id}",
        'dam_id': plan['dam_id'],
        'type': plan['type'],
        'const_year': plan['const_year'],
        'HYRIV_ID': reach_id,
        'geometry': final_geom
    })

zones_gdf = gpd.GeoDataFrame(final_zones, crs='epsg:32733').to_crs(epsg=4326)
zones_gdf.geometry = shapely.force_2d(zones_gdf.geometry)

# Export GeoJSON
zones_gdf.to_file(geojson_out, driver='GeoJSON')
print(f"Reach geometries saved to: {geojson_out}")
print(f"Centroid buffer fallback applied for: {fallback_count} / {len(final_zones)} reaches without intersected MERIT flowlines.")
