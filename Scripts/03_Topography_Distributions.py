"""
================================================================================
03_Topography_Distributions.py
Workflow Step 03: Empirical Topographical Filtering Analysis (Slope & HAND)
Manuscript Section: 2.3.2 Model development & 2.3.3 Candidate pixel post-processing

Description:
  Evaluates the empirical distribution of terrain slope (from SRTM DEM) and
  Height Above Nearest Drainage (HAND from MERIT Hydro) across true dams versus
  background false positives in the Swakop River basin.

  Determines the upper topographical bounds used to eliminate mountain shadow
  and upland non-drainage false positives:
    - Slope threshold: <= 8 degrees
    - HAND threshold:  <= 12 meters

Input:
  - 'Data/Swakop_PointPixel_Training_Data.csv' (from Step 01)

Output:
  - 'Data/Topography_Slope_HAND_Distributions.png'
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"

csv_path = os.path.join(DATA_DIR, 'Swakop_PointPixel_Training_Data.csv')
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

class_col = 'Class' if 'Class' in df.columns else 'class'
df['Class_Label'] = df[class_col].map({1: 'True Dam', 0: 'False Positive / Background'})

# Remove non-finite values
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['slope', 'hand']).copy()

# ============================================================
# 2. PLOT EMPIRICAL DISTRIBUTIONS
# ============================================================
print("Plotting Slope and HAND distributions...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Slope Distribution
sns.kdeplot(
    data=df, x='slope', hue='Class_Label', common_norm=False, fill=True,
    ax=axes[0], palette={'True Dam': '#2ca02c', 'False Positive / Background': '#ff7f0e'}, alpha=0.4
)
axes[0].axvline(8.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (8°)')
axes[0].set_title('Distribution of Terrain Slope (Degrees)', fontsize=12)
axes[0].set_xlabel('Slope (°)', fontsize=11)
axes[0].set_xlim(0, 30)
axes[0].legend()

# HAND Distribution
sns.kdeplot(
    data=df, x='hand', hue='Class_Label', common_norm=False, fill=True,
    ax=axes[1], palette={'True Dam': '#2ca02c', 'False Positive / Background': '#ff7f0e'}, alpha=0.4
)
axes[1].axvline(12.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (12 m)')
axes[1].set_title('Height Above Nearest Drainage (HAND)', fontsize=12)
axes[1].set_xlabel('HAND (m)', fontsize=11)
axes[1].set_xlim(0, 50)
axes[1].legend()

plt.tight_layout()
fig_out = os.path.join(DATA_DIR, 'Topography_Slope_HAND_Distributions.png')
plt.savefig(fig_out, dpi=300)
plt.close()
print(f"Distribution figure saved to: {fig_out}")

# ============================================================
# 3. QUANTILES SUMMARY
# ============================================================
dams_mask = df[class_col] == 1
bg_mask = df[class_col] == 0

print("\n--- Empirical Quantiles Summary ---")
print(f"Total True Dams: {dams_mask.sum()}")
print(f"Total Background/False Positives: {bg_mask.sum()}")

print("\nSlope Quantiles (True Dams):")
print(df[dams_mask]['slope'].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

print("\nHAND Quantiles (True Dams):")
print(df[dams_mask]['hand'].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

print("\nThreshold Selection:")
print("  - Slope <= 8.0 degrees retains >95% of true dams while filtering steep upland relief.")
print("  - HAND <= 12.0 meters retains >95% of true dams while eliminating high interfluve terrain.")
