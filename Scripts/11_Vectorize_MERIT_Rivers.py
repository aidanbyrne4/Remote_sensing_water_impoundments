"""
================================================================================
11_Vectorize_MERIT_Rivers.py
Workflow Step 11: High-Resolution MERIT Hydro Drainage Vectorization
Manuscript Section: 2.5.2 Vegetation and rainfall data

Description:
  Extracts and vectorizes the high-resolution MERIT Hydro drainage network
  (upstream contributing area UPA > 10 km²) across the Ugab River basin.

  The vectorized MERIT flowlines are used in Step 12 to geographically align
  and correct the 120-m river corridor reach geometries, as MERIT Hydro more
  accurately traces ephemeral braided river courses than the generalized
  HydroRIVERS vector centerline.

Inputs:
  - 'Data/MERIT_Hydro_UPA_Namibia.tif' (MERIT Hydro upstream contributing area raster)
  - 'Data/Ugab_subbasin_shapefile.shp'

Outputs:
  - 'Data/MERIT_Hydro_Rivers_10_Ugab.shp' (Used by Step 12 for corridor reach refinement)
================================================================================
"""

import os
import warnings
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import geopandas as gpd
import numpy as np

warnings.filterwarnings('ignore')

# ============================================================
# 1. FILE CONFIGURATIONS
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

tif_path = os.path.join(DATA_DIR, 'Ugab_MERIT_Hydro_UPA.tif')
basin_path = os.path.join(DATA_DIR, 'Ugab_Basin_Boundary.gpkg')
out_shp = os.path.join(DATA_DIR, 'MERIT_Hydro_Rivers_10_Ugab.shp')

# ============================================================
# 2. MASK & THRESHOLD MERIT HYDRO RASTER
# ============================================================
print(f"Loading basin geometry from: {basin_path}")
basin = gpd.read_file(basin_path)

print(f"Loading and masking MERIT Hydro UPA raster from: {tif_path}")
with rasterio.open(tif_path) as src:
    basin_matched = basin.to_crs(src.crs)
    basin_geom = [basin_matched.geometry.iloc[0]]
    
    out_image, out_transform = mask(src, basin_geom, crop=True)
    upa = out_image[0]
    
    # Clean nodata values and apply upstream area threshold (UPA > 10 km²)
    print("Filtering drainage lines: Upstream Contributing Area (UPA) > 10 km²...")
    upa = np.where(upa < 0, 0, upa)
    mask_arr = (upa > 10).astype('uint8')
    
    # ============================================================
    # 3. VECTORIZE FLOWLINES
    # ============================================================
    print("Vectorizing raster flowlines to polygon geometry...")
    results = (
        {'properties': {'upa_thresh': 10}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(mask_arr, mask=mask_arr, transform=out_transform))
    )
    
    gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf.set_crs(src.crs, inplace=True)
    
    # Save output shapefile
    print(f"Saving vectorized MERIT flowlines to: {out_shp}")
    gdf.to_file(out_shp)

print("MERIT Hydro drainage vectorization complete!")
