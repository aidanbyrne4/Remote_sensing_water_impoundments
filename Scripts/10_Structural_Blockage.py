"""
================================================================================
10_Structural_Blockage.py
Workflow Step 10: Drainage Network Snapping & Cumulative Blockage Modeling
Manuscript Section: 2.5.1 Drainage-network analysis

Description:
  1. Links verified artificial water impoundments (from Step 07) to the nearest
     tributary in the HydroRIVERS network using a maximum snapping distance of 5 km.
  2. Constructs a directed drainage network using the HydroRIVERS NEXT_DOWN
     connectivity matrix.
  3. Traverses the directed graph downstream to accumulate total upstream
     impoundment surface area (cum_dam_km2) for every river segment.
  4. Calculates cumulative upstream impoundment burden as total upstream dam area
     divided by upstream catchment area (cum_dam_km2 / UPLAND_SKM).

Inputs:
  - 'Data/Ugab_Cleaned_Dams_Final.shp' (from Step 07)
  - 'Data/HydroRivers_Nambia_west.shp' (HydroRIVERS dataset)
  - 'Data/Ugab_subbasin_shapefile.shp'

Outputs:
  - 'Data/Ugab_Dams_Snapped.csv'
  - 'Data/Ugab_Rivers_Cumulative_Blockage.shp' (Crucial river network for Steps 12, 15, 16)
  - 'Data/Structural_Blockage_Map.png'
================================================================================
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# 1. FILE CONFIGURATIONS & LOADING
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

# Input files in Data directory
dams_path = os.path.join(DATA_DIR, 'Ugab_Cleaned_Dams_Final.shp')
rivers_path = os.path.join(DATA_DIR, 'Ugab_HydroRIVERS_Network.gpkg')
basin_path = os.path.join(DATA_DIR, 'Ugab_Basin_Boundary.gpkg')

# Output files in Data directory
out_csv = os.path.join(DATA_DIR, 'Ugab_Dams_Snapped.csv')
out_shp = os.path.join(DATA_DIR, 'Ugab_Rivers_Cumulative_Blockage.shp')
out_fig = os.path.join(DATA_DIR, 'Structural_Blockage_Map.png')

print(f"Loading dams from: {dams_path}")
dams = gpd.read_file(dams_path)

print(f"Loading HydroRIVERS network from: {rivers_path}")
rivers = gpd.read_file(rivers_path)
basin = gpd.read_file(basin_path)

# Reproject to UTM Zone 33S (EPSG:32733) for accurate metric distance and area calculations
dams = dams.to_crs(epsg=32733)
rivers = rivers.to_crs(epsg=32733)
basin = basin.to_crs(epsg=32733)

# Clip river network to the Ugab catchment boundary
rivers = gpd.clip(rivers, basin).reset_index(drop=True)
rivers['riv_idx'] = rivers.index

# Dam surface area in km²
dams['dam_area_km2'] = dams.geometry.area / 1e6

# ============================================================
# 2. SNAP DAMS TO HYDRORIVERS (MAX 5 KM DISTANCE)
# ============================================================
print("Snapping dams to nearest HydroRIVERS reach (max distance = 5,000 m)...")
snapped = gpd.sjoin_nearest(dams, rivers, how='left', max_distance=5000, distance_col='snap_dist')

# Tie-break equidistant reaches by closest distance, then by smallest contributing catchment area
id_col_left = 'id_left' if 'id_left' in snapped.columns else 'id'
snapped = snapped.sort_values(['snap_dist', 'UPLAND_SKM'], ascending=[True, True]).drop_duplicates(subset=[id_col_left])

def get_confidence(dist):
    if dist <= 500: return 'High'
    elif dist <= 1000: return 'Medium'
    else: return 'Low'

snapped['snap_conf'] = snapped['snap_dist'].apply(get_confidence)
snapped_count = len(snapped[snapped['HYRIV_ID'].notna()])
print(f"Successfully snapped {snapped_count} / {len(dams)} impoundments to the drainage network.")

# Save snapped dams table
df_snapped = pd.DataFrame(snapped.drop(columns='geometry'))
df_snapped.to_csv(out_csv, index=False)
print(f"Snapped dams table saved to: {out_csv}")

# ============================================================
# 3. BUILD DIRECTED GRAPH & ACCUMULATE CUMULATIVE BURDEN
# ============================================================
print("Building directed network topology and routing upstream blockage...")
G = nx.DiGraph()

# Add river reaches as directed edges (flow from HYRIV_ID -> NEXT_DOWN)
for _, row in rivers.iterrows():
    G.add_edge(row['HYRIV_ID'], row['NEXT_DOWN'])
    G.nodes[row['HYRIV_ID']]['local_dam_area'] = 0.0

# Assign local dam surface area to corresponding reaches
for _, row in snapped.iterrows():
    if pd.notna(row['HYRIV_ID']) and row['HYRIV_ID'] in G.nodes:
        G.nodes[row['HYRIV_ID']]['local_dam_area'] += row['dam_area_km2']

# Accumulate dam surface area downstream through graph ancestors
cum_areas = {}
for node in G.nodes():
    try:
        ancestors = nx.ancestors(G, node)
        cum_area = G.nodes[node]['local_dam_area'] + sum(G.nodes[a].get('local_dam_area', 0.0) for a in ancestors)
        cum_areas[node] = cum_area
    except Exception:
        cum_areas[node] = G.nodes[node].get('local_dam_area', 0.0)

rivers['cum_dam_km2'] = rivers['HYRIV_ID'].map(cum_areas)

# Calculate Blockage Ratio: Cumulative Dam Area / Upstream Catchment Area (dimensionless)
rivers['block_ratio'] = rivers['cum_dam_km2'] / rivers['UPLAND_SKM']
rivers['block_ratio'] = rivers['block_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

# Save standardized shapefile
print(f"Saving cumulative blockage river network to: {out_shp}")
rivers.to_file(out_shp)

# ============================================================
# 4. GENERATE STRUCTURAL BLOCKAGE MAP
# ============================================================
print("Generating structural blockage map...")
fig, ax = plt.subplots(figsize=(10, 6.5))
basin.to_crs(epsg=4326).boundary.plot(ax=ax, color='black', linewidth=1, label='Ugab Basin')

rivers_wgs = rivers.to_crs(epsg=4326)
rivers_wgs[rivers_wgs['block_ratio'] == 0].plot(ax=ax, color='#a6cee3', linewidth=0.7, label='Pristine Network')
rivers_wgs[rivers_wgs['block_ratio'] > 0].plot(
    ax=ax, column='block_ratio', cmap='YlOrRd', linewidth=1.5,
    legend=True, legend_kwds={'label': 'Blockage Ratio (Dam Area / Catchment Area)'}
)

dams_wgs = dams.to_crs(epsg=4326)
dams_wgs.plot(ax=ax, color='blue', markersize=6, alpha=0.6, label='Snapped Impoundments')

ax.set_title('Ugab River Basin: Impoundment Distribution and Upstream Structural Blockage', fontsize=12)
ax.set_xlabel('Longitude (°E)', fontsize=10)
ax.set_ylabel('Latitude (°S)', fontsize=10)
plt.tight_layout()
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"Structural blockage figure saved to: {out_fig}")
