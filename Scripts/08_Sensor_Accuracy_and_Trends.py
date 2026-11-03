"""
================================================================================
08_Sensor_Accuracy_and_Trends.py
Workflow Step 08: Sensor Accuracy Evaluation & Historical Construction Trends
Manuscript Section: 2.4 Impoundment construction trends

Description:
  1. Filters the 1990-2026 Landsat time series to the 758 verified active dams
     identified in Step 07 and exports the cleaned time-series dataset.
  2. Evaluates cross-sensor consistency between 30-m Landsat and 10-m Sentinel-2
     during the overlapping observation period (2016-2026):
     - Linear agreement (Pearson correlation)
     - Overall recall and precision
     - Detection recall stratified across dam size classes (<1000m², 1000-5000m², >5000m²)
  3. Estimates the first remotely detectable establishment/filling year for every
     impoundment based on the satellite criterion (annual max MNDWI > 0.05).
  4. Generates the decadal establishment summary and construction timeline figure.

Inputs:
  - 'Data/Ugab_Dams_S2_TimeSeries_2016_2026.csv' (from Step 06)
  - 'Data/Ugab_Dams_Landsat_TimeSeries_1990_2026.csv' (from Step 06)
  - 'Data/Ugab_Cleaned_Dams_Final.shp' (from Step 07)

Outputs:
  - 'Data/Ugab_Dams_Landsat_Cleaned_TimeSeries.csv' (Used by Steps 09, 12, 15, 16)
  - 'Data/Dam_Construction_Timeline.png'
================================================================================
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. FILE CONFIGURATIONS & LOADING
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

s2_csv = os.path.join(DATA_DIR, 'Ugab_Dams_S2_TimeSeries_2016_2026.csv')
l8_csv = os.path.join(DATA_DIR, 'Ugab_Dams_Landsat_TimeSeries_1990_2026.csv')
shp_path = os.path.join(DATA_DIR, 'Ugab_Cleaned_Dams_Final.shp')

out_csv = os.path.join(DATA_DIR, 'Ugab_Dams_Landsat_Cleaned_TimeSeries.csv')
plot_path = os.path.join(DATA_DIR, 'Dam_Construction_Timeline.png')

print(f"Loading Sentinel-2 data from: {s2_csv}")
df_s2 = pd.read_csv(s2_csv)

print(f"Loading Landsat data from: {l8_csv}")
df_l8 = pd.read_csv(l8_csv)

print(f"Loading cleaned dams from: {shp_path}")
gdf = gpd.read_file(shp_path)

id_col_gdf = 'id' if 'id' in gdf.columns else 'FID'
valid_ids = gdf[id_col_gdf].tolist()
print(f"Filtering time series to {len(valid_ids)} verified active impoundments...")

df_s2.rename(columns={'id' if 'id' in df_s2.columns else 'system:index': 'id'}, inplace=True)
df_l8.rename(columns={'id' if 'id' in df_l8.columns else 'system:index': 'id'}, inplace=True)

df_s2 = df_s2[df_s2['id'].isin(valid_ids)].copy()
df_l8 = df_l8[df_l8['id'].isin(valid_ids)].copy()

# ============================================================
# 2. SAVE CLEANED LANDSAT TIME SERIES (STEP 08 CORE OUTPUT)
# ============================================================
print(f"Saving cleaned Landsat time series to: {out_csv}")
df_l8.to_csv(out_csv, index=False)

mndwi_s2 = [c for c in df_s2.columns if 'MNDWI_max' in c or 'MNDWI' in c][0]
mndwi_l8 = [c for c in df_l8.columns if 'MNDWI_max' in c or 'MNDWI' in c][0]

# ============================================================
# 3. SENSOR ACCURACY EVALUATION (LANDSAT VS. SENTINEL-2, 2016-2026)
# ============================================================
s2_overlap = df_s2[df_s2['Year'].between(2016, 2026)].copy()
l8_overlap = df_l8[df_l8['Year'].between(2016, 2026)].copy()

overlap = s2_overlap.merge(l8_overlap, on=['id', 'Year'], suffixes=('_s2', '_l8'))
overlap['s2_has_water'] = overlap[mndwi_s2 + '_s2'] > 0.0
overlap['l8_has_water'] = overlap[mndwi_l8 + '_l8'] > 0.0

tp = len(overlap[(overlap['s2_has_water']) & (overlap['l8_has_water'])])
fn = len(overlap[(overlap['s2_has_water']) & (~overlap['l8_has_water'])])
fp = len(overlap[(~overlap['s2_has_water']) & (overlap['l8_has_water'])])

print("\n--- Landsat vs. Sentinel-2 Accuracy (2016-2026 Overlap) ---")
print(f"Paired Inundation Observations: True Positives={tp}, False Negatives={fn}, False Positives={fp}")
if (tp + fp) > 0: print(f"Precision: {tp / (tp + fp):.3f}")
if (tp + fn) > 0: print(f"Recall (Sensitivity): {tp / (tp + fn):.3f}")

# Cross-sensor Pearson correlation
r = overlap[mndwi_s2 + '_s2'].corr(overlap[mndwi_l8 + '_l8'])
print(f"Pearson Correlation (S2 vs Landsat Peak MNDWI): r = {r:.3f}")

# Recall by Dam Surface Area Size Class
gdf_proj = gdf.to_crs('EPSG:32733')
gdf_proj['area_m2'] = gdf_proj.geometry.area
overlap = overlap.merge(gdf_proj[[id_col_gdf, 'area_m2']].rename(columns={id_col_gdf: 'id'}), on='id')
overlap['size_class'] = pd.cut(
    overlap['area_m2'],
    bins=[0, 1000, 5000, np.inf],
    labels=['Small (<1000m²)', 'Medium (1000-5000m²)', 'Large (>5000m²)']
)

print("\n--- Landsat Inundation Recall by Dam Size Class ---")
for sz in ['Small (<1000m²)', 'Medium (1000-5000m²)', 'Large (>5000m²)']:
    sub = overlap[overlap['size_class'] == sz]
    stp = len(sub[(sub['s2_has_water']) & (sub['l8_has_water'])])
    sfn = len(sub[(sub['s2_has_water']) & (~sub['l8_has_water'])])
    rec_val = stp / (stp + sfn) if (stp + sfn) > 0 else 0
    print(f"  {sz}: {rec_val:.3f} (N = {len(sub['id'].unique())} dams)")

# ============================================================
# 4. CONSTRUCTION TRENDS & ESTABLISHMENT TIMELINE
# ============================================================
print("\n--- Impoundment Establishment Timeline (1990-2026) ---")

# First year exceeding conservative remote sensing establishment threshold (MNDWI > 0.05)
def get_construction_year(df, mndwi_col, start_year=1990, threshold=0.05):
    res = []
    for dam_id, group in df.groupby('id'):
        wy = group[group[mndwi_col] > threshold].sort_values('Year')['Year'].values
        if len(wy) > 0:
            yr = int(wy[0])
            status = 'Built Before 1990' if yr == start_year else 'Observed Construction'
            res.append({'id': dam_id, 'yr': yr, 'status': status})
        else:
            res.append({'id': dam_id, 'yr': -1, 'status': 'Never Exceeded Threshold'})
    return pd.DataFrame(res)

const_l8 = get_construction_year(df_l8, mndwi_l8, start_year=1990, threshold=0.05)

built_before_90 = len(const_l8[const_l8['yr'] == 1990])
const_l8['decade'] = pd.cut(
    const_l8['yr'],
    bins=[1990, 1999, 2009, 2019, 2029],
    labels=['1990s', '2000s', '2010s', '2020s']
)

print(f"Dams established on or before 1990: {built_before_90}")
print("Dams constructed by decade (1991-2026):")
print(const_l8[const_l8['status'] == 'Observed Construction']['decade'].value_counts().sort_index().to_string())

counts = const_l8[const_l8['yr'] >= 1990]['yr'].value_counts().sort_index()
counts = counts.reindex(range(1990, 2027), fill_value=0)

# ============================================================
# 5. GENERATE BROKEN-AXIS TIMELINE FIGURE
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 5.5), gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.08)

ax1.bar(counts.index, counts.values, width=0.85, color='#2b5c8f', edgecolor='white')
ax2.bar(counts.index, counts.values, width=0.85, color='#2b5c8f', edgecolor='white')

max_lower = counts[counts.index > 1990].max() + 5
ax2.set_ylim(0, max_lower)
ax1.set_ylim(counts[1990] - 25, counts[1990] + 25)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d*3, +d*3), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

ax2.set_xlabel('Year of Initial Dam Filling / Establishment', fontsize=11)
fig.text(0.06, 0.5, 'Number of Impoundments', va='center', rotation='vertical', fontsize=11)
ax1.set_title('Ugab Basin: Dam Establishment Timeline (<=1990 - 2026)', pad=15, fontsize=13)

ticks = range(1990, 2027, 4)
ax2.set_xticks(ticks)
ax2.set_xticklabels([str(t) if t != 1990 else '<=1990' for t in ticks])

# Cumulative curve on secondary axis
plt.draw()
pos1, pos2 = ax1.get_position(), ax2.get_position()
ax3 = fig.add_axes([pos2.x0, pos2.y0, pos2.width, pos1.y1 - pos2.y0])
ax3.patch.set_alpha(0)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.spines['left'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.xaxis.set_visible(False)

cum_counts = counts.cumsum()
cum_pct = (cum_counts / cum_counts.max()) * 100
ax3.set_xlim(ax2.get_xlim())
ax3.plot(cum_pct.index, cum_pct.values, color='dimgrey', marker='o', linewidth=2, markersize=3.5, label='Cumulative %')
ax3.set_ylim(0, 105)
ax3.set_ylabel('Cumulative Percentage (%)', color='dimgrey', rotation=270, labelpad=18, fontsize=11)

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Construction timeline figure successfully saved to: {plot_path}")
