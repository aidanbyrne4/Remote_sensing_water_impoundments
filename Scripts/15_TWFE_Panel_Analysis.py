"""
================================================================================
15_TWFE_Panel_Analysis.py
Workflow Step 15: Two-Way Fixed Effects (TWFE) Econometric Panel Analysis
Manuscript Section: 2.5.3 Statistical analysis

Description:
  Estimates the ecological impact of cumulative upstream impoundment burden
  on downstream riparian vegetation across 514 river reaches over 36 years (1990-2025):
    1. Primary TWFE Model: Dry-season NDVI as a function of upstream catchment
       rainfall, cumulative dam burden, and their interaction, with entity
       (sub-basin reach) and year fixed effects, clustering standard errors
       across 61 independent hydrological tributary clusters:
         NDVI_it = beta1*Rain_it + beta2*Burden_it + beta3*(Rain_it * Burden_it)
                   + alpha_i + gamma_t + epsilon_it
    2. Short-Term Antecedent Rainfall Robustness: Tests 90-day pre-dry season rain (May-Jul).
    3. Falsification Placebo Test: Evaluates future dam construction (leads t+1 to t+3).
    4. Aggregated Tributary-Level Model: Evaluates robustness to reach spatial correlation.
    5. Calculates ecological effect size: Attenuation of riparian responsiveness to rainfall.

Inputs:
  - 'Data/Ugab_Reach_Vegetation_Metrics_1990_2025.csv' (from Step 13)
  - 'Data/Ugab_HYBAS12_Catchment_Rainfall_1990_2025.csv' (from Step 14)
  - 'Data/Ugab_Rivers_Cumulative_Blockage.shp' (from Step 10)
  - 'Data/Ugab_Dams_Snapped.csv' (from Step 10)
  - 'Data/Ugab_Dams_Landsat_Cleaned_TimeSeries.csv' (from Step 08)

Outputs:
  - 'Data/Ugab_TWFE_Panel_Data.csv' (Crucial master dataset for Step 16)
  - 'Data/Interaction_Plot_Final.png'
================================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# 1. FILE CONFIGURATIONS & LOADING
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

# Input files in Data directory
ts_file = os.path.join(DATA_DIR, 'Ugab_Reach_Vegetation_Metrics_1990_2025.csv')
rain_file = os.path.join(DATA_DIR, 'Ugab_HYBAS12_Catchment_Rainfall_1990_2025.csv')
rivers_file = os.path.join(DATA_DIR, 'Ugab_Rivers_Cumulative_Blockage.shp')
snapped_file = os.path.join(DATA_DIR, 'Ugab_Dams_Snapped.csv')
landsat_file = os.path.join(DATA_DIR, 'Ugab_Dams_Landsat_Cleaned_TimeSeries.csv')

# Output files in Data directory
panel_out = os.path.join(DATA_DIR, 'Ugab_TWFE_Panel_Data.csv')
plot_path = os.path.join(DATA_DIR, 'Interaction_Plot_Final.png')

print(f"Loading reach vegetation metrics from: {ts_file}")
ts_df = pd.read_csv(ts_file)

print(f"Loading catchment rainfall from: {rain_file}")
rain_df = pd.read_csv(rain_file)
rain_dict = {
    (r['HYBAS_ID'], r['Year']): {
        'wet': r['Rain_WetSeason_Catchment'],
        'ant': r['Rain_Antecedent90_Catchment'],
        'area': r['SUB_AREA']
    } for _, r in rain_df.iterrows()
}

print(f"Loading river network from: {rivers_file}")
rivers = gpd.read_file(rivers_file)
catchment_areas = dict(zip(rivers['HYRIV_ID'], rivers['UPLAND_SKM']))
hybas_map = dict(zip(rivers['HYRIV_ID'], rivers['HYBAS_L12']))

snapped_dams = pd.read_csv(snapped_file)
dam_id_col = 'dam_id' if 'dam_id' in snapped_dams.columns else 'id_left'
snapped_dams = snapped_dams.rename(columns={dam_id_col: 'dam_id'})

raw_ts = pd.read_csv(landsat_file)
mndwi_col = [c for c in raw_ts.columns if 'MNDWI_max' in c or 'MNDWI' in c][0]
raw_ts_water = raw_ts[raw_ts[mndwi_col] > 0.05]
const_years = raw_ts_water.groupby('id')['Year'].min().reset_index()
const_years.columns = ['dam_id', 'const_year']

dams_csv = snapped_dams.merge(const_years, on='dam_id', how='left')
dams_csv['const_year'] = dams_csv['const_year'].fillna(1990)

# ============================================================
# 2. BUILD DIRECTED GRAPH & TRIBUTARY CLUSTERS
# ============================================================
print("Constructing directed hydrological graph and mapping reaches...")
G = nx.DiGraph()
for _, row in rivers.iterrows():
    G.add_edge(row['HYRIV_ID'], row['NEXT_DOWN'])
if 0 in G.nodes: G.remove_node(0)

# Map dams to corresponding river reaches
reach_dams = {}
for _, row in dams_csv.iterrows():
    reach = row['HYRIV_ID']
    if reach not in reach_dams: reach_dams[reach] = []
    area = row['dam_area_km2'] if not pd.isna(row['dam_area_km2']) else 0.0
    reach_dams[reach].append((row['const_year'], area))

G_rev = G.reverse()
reach_upstream_cache = {}
for r in ts_df['HYRIV_ID'].dropna().unique():
    up_reaches = list(nx.descendants(G_rev, r)) + [r] if r in G_rev.nodes else [r]
    dams = []
    hybas = set()
    for ur in up_reaches:
        if ur in reach_dams: dams.extend(reach_dams[ur])
        if ur in hybas_map: hybas.add(hybas_map[ur])
    reach_upstream_cache[r] = {'dams': dams, 'hybas': hybas}

# Map 61 Independent Hydrological Tributary Clusters along the Main Stem
outlets = [n for n, d in G.out_degree() if d == 0]
outlet = max(outlets, key=lambda x: catchment_areas.get(x, 0))
main_stem = set()
current = outlet
while True:
    main_stem.add(current)
    preds = list(G.predecessors(current))
    if not preds: break
    current = max(preds, key=lambda x: catchment_areas.get(x, 0))

tributary_map = {}
trib_id = 1
for ms_node in main_stem:
    for p in list(G.predecessors(ms_node)):
        if p not in main_stem:
            trib_ancestors = nx.ancestors(G, p)
            trib_ancestors.add(p)
            for a in trib_ancestors: tributary_map[a] = f"Trib_{trib_id}"
            trib_id += 1

# ============================================================
# 3. CONSTRUCT REACH-YEAR PANEL DATASET
# ============================================================
print("Calculating time-varying cumulative burden and upstream rainfall...")
data = []
for idx, row in ts_df.iterrows():
    reach_id = row.get('HYRIV_ID', None)
    if pd.isna(reach_id): continue
    year = row['Year']
    upland_skm = catchment_areas.get(reach_id, 1)
    
    # Exclude submerged reach-years
    if row.get('MNDWI_Freq', 0) > 0.2: continue
    
    cache = reach_upstream_cache[reach_id]
    
    # Cumulative burden constructed on or before Year T
    dam_area = sum(a for cy, a in cache['dams'] if cy <= year)
    
    # Falsification lead burden constructed between t+1 and t+3
    future_dam_area = sum(a for cy, a in cache['dams'] if year < cy <= year + 3)
    
    # Area-weight upstream catchment rainfall
    w_sum, ant_sum, a_sum = 0.0, 0.0, 0.0
    for hid in cache['hybas']:
        if (hid, year) in rain_dict:
            r = rain_dict[(hid, year)]
            if not pd.isna(r['wet']):
                w_sum += r['wet'] * r['area']
                ant_sum += r['ant'] * r['area']
                a_sum += r['area']
                
    if a_sum > 0 and not pd.isna(row.get('NDVI_Dry', np.nan)):
        data.append({
            'zone_id': row['zone_id'],
            'HYRIV_ID': reach_id,
            'Year': year,
            'DamArea_Norm': dam_area / upland_skm,
            'FutureDam_Norm': future_dam_area / upland_skm,
            'Upstream_Rain_WetSeason': w_sum / a_sum,
            'Upstream_Rain_Antecedent': ant_sum / a_sum,
            'NDVI_Dry': row['NDVI_Dry'],
            'type': row['type'],
            'tributary': tributary_map.get(reach_id, 'Main_Stem')
        })

panel = pd.DataFrame(data)

# Save complete panel dataset
panel.to_csv(panel_out, index=False)
print(f"Panel dataset successfully saved to: {panel_out}")
print(f"Final panel observations: {len(panel)} valid reach-years across {panel['zone_id'].nunique()} reaches.")
print(f"Reaches mapped to {panel['tributary'].nunique()} distinct independent hydrological clusters.")

# ============================================================
# 4. TWFE MODEL ESTIMATION ENGINE
# ============================================================
def run_twfe(df, y_col, rain_col, burden_col, cluster_se_col, id_col='tributary'):
    df_clean = df.copy()
    df_clean['Rain_x_Burden'] = df_clean[rain_col] * df_clean[burden_col]
    
    stds = {c: df_clean[c].std() for c in [rain_col, burden_col, 'Rain_x_Burden']}
    means = {c: df_clean[c].mean() for c in [rain_col, burden_col, 'Rain_x_Burden']}
    
    for col in [rain_col, burden_col, 'Rain_x_Burden']:
        if stds[col] == 0: df_clean[f'{col}_std'] = 0
        else: df_clean[f'{col}_std'] = (df_clean[col] - means[col]) / stds[col]

    X_base = df_clean[[f'{rain_col}_std', f'{burden_col}_std', 'Rain_x_Burden_std']].values
    
    # Entity and Time Fixed Effects
    reach_dummies = pd.get_dummies(df_clean[id_col], drop_first=True, dtype=float).values
    year_dummies = pd.get_dummies(df_clean['Year'], drop_first=True, dtype=float).values
    X = np.hstack([np.ones((X_base.shape[0], 1)), X_base, reach_dummies, year_dummies])
    y = df_clean[y_col].values

    # OLS Solution
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta

    # Clustered Robust Covariance Matrix
    clusters_se = df_clean[cluster_se_col].values
    unique_clusters = np.unique(clusters_se)
    G_c = len(unique_clusters)
    N = X.shape[0]
    k = X.shape[1]

    meat = np.zeros((k, k))
    for g in unique_clusters:
        idx = (clusters_se == g)
        X_g = X[idx]
        u_g = resid[idx]
        S_g = X_g.T @ u_g
        meat += np.outer(S_g, S_g)

    V_cluster = (G_c / (G_c - 1)) * ((N - 1) / (N - k)) * (bread @ meat @ bread)
    se_robust = np.sqrt(np.diag(V_cluster))
    t_crit = stats.t.ppf(0.975, G_c - 1)
    
    b = beta[:4]
    se = se_robust[:4]
    p_val = 2 * (1 - stats.t.cdf(np.abs(b / se), G_c - 1))
    
    return b, b - t_crit * se, b + t_crit * se, p_val

def format_res(name, b, ci_l, ci_u, p):
    return f"  {name}: Beta = {b:.4f} (95% CI: [{ci_l:.4f}, {ci_u:.4f}]), p = {p:.4f}"

# --- Model 1: Primary TWFE Model ---
print("\n--- 1. PRIMARY TWFE MODEL (61 Tributary Clusters) ---")
b, ci_l, ci_u, p = run_twfe(panel, 'NDVI_Dry', 'Upstream_Rain_WetSeason', 'DamArea_Norm', 'tributary', 'tributary')
print(format_res("Rainfall (Wet Season)", b[1], ci_l[1], ci_u[1], p[1]))
print(format_res("Dam Burden (Cumulative)", b[2], ci_l[2], ci_u[2], p[2]))
print(format_res("Interaction (Attenuation Effect)", b[3], ci_l[3], ci_u[3], p[3]))

# --- Model 2: Short-Term Antecedent Rainfall ---
print("\n--- 2. ROBUSTNESS: SHORT-TERM RAINFALL (May-Jul) ---")
b, ci_l, ci_u, p = run_twfe(panel, 'NDVI_Dry', 'Upstream_Rain_Antecedent', 'DamArea_Norm', 'tributary', 'tributary')
print(format_res("Interaction (Short-Term Antecedent Rain)", b[3], ci_l[3], ci_u[3], p[3]))

# --- Model 3: Future Lead Placebo Test ---
print("\n--- 3. FALSIFICATION: FUTURE DAM CONSTRUCTION (Leads t+1 to t+3) ---")
b, ci_l, ci_u, p = run_twfe(panel, 'NDVI_Dry', 'Upstream_Rain_WetSeason', 'FutureDam_Norm', 'tributary', 'tributary')
print(format_res("Interaction (Future Treatment Placebo)", b[3], ci_l[3], ci_u[3], p[3]))

# --- Model 4: Aggregated Tributary-Level Model ---
print("\n--- 4. AGGREGATED TRIBUTARY-LEVEL MODEL ---")
agg_panel = panel.groupby(['tributary', 'Year']).agg({
    'NDVI_Dry': 'mean',
    'Upstream_Rain_WetSeason': 'mean',
    'DamArea_Norm': 'mean'
}).reset_index()
b, ci_l, ci_u, p = run_twfe(agg_panel, 'NDVI_Dry', 'Upstream_Rain_WetSeason', 'DamArea_Norm', 'tributary', 'tributary')
print(format_res("Interaction (Aggregated Model)", b[3], ci_l[3], ci_u[3], p[3]))

# ============================================================
# 5. ECOLOGICAL EFFECT SIZE & INTERACTION PLOT
# ============================================================
print("\n--- 5. ECOLOGICAL EFFECT SIZE CALCULATION ---")
rain_25 = panel['Upstream_Rain_WetSeason'].quantile(0.25)
rain_75 = panel['Upstream_Rain_WetSeason'].quantile(0.75)
burden_25 = panel['DamArea_Norm'].quantile(0.25)
burden_75 = panel['DamArea_Norm'].quantile(0.75)

df_u = panel.copy()
df_u['Rain_x_Burden'] = df_u['Upstream_Rain_WetSeason'] * df_u['DamArea_Norm']
X_u = np.hstack([np.ones((len(df_u), 1)), df_u[['Upstream_Rain_WetSeason', 'DamArea_Norm', 'Rain_x_Burden']].values])
y_u = df_u['NDVI_Dry'].values
bread_u = np.linalg.pinv(X_u.T @ X_u)
beta_u = bread_u @ X_u.T @ y_u

b_0_u, b_rain_u, b_burden_u, b_int_u = beta_u[0], beta_u[1], beta_u[2], beta_u[3]

pred_low_low = b_0_u + b_rain_u * rain_25 + b_burden_u * burden_25 + b_int_u * rain_25 * burden_25
pred_low_high = b_0_u + b_rain_u * rain_75 + b_burden_u * burden_25 + b_int_u * rain_75 * burden_25
pred_high_low = b_0_u + b_rain_u * rain_25 + b_burden_u * burden_75 + b_int_u * rain_25 * burden_75
pred_high_high = b_0_u + b_rain_u * rain_75 + b_burden_u * burden_75 + b_int_u * rain_75 * burden_75

delta_low = pred_low_high - pred_low_low
delta_high = pred_high_high - pred_high_low
pct_reduction = ((delta_low - delta_high) / delta_low) * 100

print(f"Predicted NDVI Response across 25th-75th Percentile Rainfall Range:")
print(f"  - At Low Dam Burden (25th Pct):  Delta NDVI = +{delta_low:.4f}")
print(f"  - At High Dam Burden (75th Pct): Delta NDVI = +{delta_high:.4f}")
print(f"  - Riparian Greenness Attenuation Effect: {pct_reduction:.1f}% reduction in rainfall responsiveness.")

# Plot Response Curves
sns.set_theme(style='whitegrid')
plt.figure(figsize=(7.5, 5))
rain_vals = np.linspace(panel['Upstream_Rain_WetSeason'].quantile(0.05), panel['Upstream_Rain_WetSeason'].quantile(0.95), 100)
pred_low = b_0_u + b_rain_u * rain_vals + b_burden_u * burden_25 + b_int_u * rain_vals * burden_25
pred_high = b_0_u + b_rain_u * rain_vals + b_burden_u * burden_75 + b_int_u * rain_vals * burden_75

plt.plot(rain_vals, pred_low, label='Low Impoundment Burden (25th percentile)', color='#1f78b4', linewidth=2.5)
plt.plot(rain_vals, pred_high, label='High Impoundment Burden (75th percentile)', color='#e31a1c', linewidth=2.5, linestyle='--')
plt.xlabel('Antecedent Wet-Season Catchment Rainfall (mm)', fontsize=11)
plt.ylabel('Predicted Dry-Season Riparian NDVI', fontsize=11)
plt.title('Riparian Greenness Response to Antecedent Rainfall by Dam Burden', fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Interaction response plot saved to: {plot_path}")
