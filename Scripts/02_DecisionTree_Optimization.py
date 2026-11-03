"""
================================================================================
02_DecisionTree_Optimization.py
Workflow Step 02: Decision Tree Optimization for Impoundment Detection
Manuscript Section: 2.3.2 Model development

Description:
  Trains a constrained, interpretable decision tree classifier using scikit-learn
  on the point-pixel training dataset extracted from the Swakop River basin.
  Constraints applied to prevent overfitting and allow exact translation into
  raster algebra:
    - Maximum depth = 3
    - Minimum samples per leaf = 20
    - Balanced class weights (penalizing false negatives inversely proportional to class frequency)

  Outputs the 3 optimal decision pathways:
    1. High optical water signal: MNDWI_wet_freq > 0.07 & 3.48 < VVVH_wet_median <= 9.85
    2. Weak optical water signal: 0.01 < MNDWI_wet_freq <= 0.07 & VVVH_wet_median <= 8.21
    3. Dry-season persistence:    MNDWI_wet_freq <= 0.01 & MNDWI_dry_freq > 0.01

Input:
  - 'Data/Swakop_PointPixel_Training_Data.csv' (from Step 01)
================================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"

csv_path = os.path.join(DATA_DIR, 'Swakop_PointPixel_Training_Data.csv')

print(f"Loading point-pixel training data from: {csv_path}")
df = pd.read_csv(csv_path)

# Handle class column naming robustly
class_col = 'Class' if 'Class' in df.columns else 'class'

# Essential features evaluated for the optical-SAR decision tree
features = ['MNDWI_wet_freq', 'VVVH_wet_median', 'MNDWI_dry_freq']
df = df.dropna(subset=features + [class_col]).copy()

# Binary classification: 1 = Active Dam, 0 = Non-Dam (Background or Hard Negative)
df['is_dam'] = df[class_col].apply(lambda x: 1 if x == 1 else 0)

X = df[features]
y = df['is_dam']

print(f"Training dataset size: {len(df)} samples")
print(f"  - Active Reference Dams: {sum(y == 1)}")
print(f"  - Background / Hard Negatives: {sum(y == 0)}\n")

# ============================================================
# 2. TRAIN BALANCED DECISION TREE (DEPTH = 3, MIN LEAF = 20)
# ============================================================
print("--- Training Constrained Decision Tree ---")
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42, class_weight='balanced')
clf.fit(X, y)

# Export and display the exact decision tree split rules
tree_rules = export_text(clf, feature_names=features)
print("Fitted Decision Tree Splits:")
print(tree_rules)

# ============================================================
# 3. EVALUATE THREE ALTERNATIVE PATHWAYS (RASTER ALGEBRA FORM)
# ============================================================
print("\n--- Evaluating Three Alternative Detection Pathways ---")

def apply_decision_rules(row):
    # Pathway 1: High optical water signal (relaxed radar constraint)
    cond1 = (row['MNDWI_wet_freq'] > 0.07) and (3.48 < row['VVVH_wet_median'] <= 9.85)
    
    # Pathway 2: Weak optical water signal (strict radar constraint required)
    cond2 = (0.01 < row['MNDWI_wet_freq'] <= 0.07) and (row['VVVH_wet_median'] <= 8.21)
    
    # Pathway 3: Dry-season persistence (no radar constraint required)
    cond3 = (row['MNDWI_wet_freq'] <= 0.01) and (row['MNDWI_dry_freq'] > 0.01)
    
    return 1 if (cond1 or cond2 or cond3) else 0

df['predicted_dam'] = df.apply(apply_decision_rules, axis=1)

# Point-Pixel Performance
tp = len(df[(df['is_dam'] == 1) & (df['predicted_dam'] == 1)])
fp = len(df[(df['is_dam'] == 0) & (df['predicted_dam'] == 1)])
fn = len(df[(df['is_dam'] == 1) & (df['predicted_dam'] == 0)])
tn = len(df[(df['is_dam'] == 0) & (df['predicted_dam'] == 0)])

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"True Positives (Detected Dams): {tp}")
print(f"False Positives (Initial Noise): {fp}")
print(f"False Negatives (Missed Dams): {fn}")
print(f"True Negatives (Correctly Filtered): {tn}")
print(f"Point-Pixel Recall: {recall * 100:.1f}%")
print(f"Point-Pixel Precision: {precision * 100:.1f}%")
print("\nNote: Point-pixel recall is further augmented during spatial vector post-processing (Step 05).")
print("These 3 conditions are implemented directly in GEE raster algebra in Step 04.")
