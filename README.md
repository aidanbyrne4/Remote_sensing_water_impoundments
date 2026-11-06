# Automated remote sensing of artificial water impoundments and their ecological impacts in arid ephemeral rivers

**Author:** Aidan Byrne  
**Last updated:** 03/11/2026  
**Contact:** aidan.byrne@ioz.ac.uk

---

## Introduction

This repository contains the end-to-end computational workflow and analysis scripts supporting the study *"Automated remote sensing of artificial water impoundments and their ecological impacts in arid ephemeral rivers"*. The pipeline automates the basin-wide detection of small artificial water impoundments (farm dams) across western Namibia using multi-sensor satellite imagery (Sentinel-1, Sentinel-2, Landsat), traces hydrologic connectivity and cumulative impoundment burden across drainage networks, and evaluates downstream riparian ecological impacts using multi-decadal Two-Way Fixed Effects (TWFE) panel econometric models.

---

## Repository Structure

```text
├── Scripts/                      # Sequential analysis scripts (01 to 16)
│   ├── 01_GEE_PointPixel_Extraction.py
│   ├── 02_DecisionTree_Optimization.py
│   ├── 03_Topography_Distributions.py
│   ├── 04_GEE_Dam_Detection_Export.py
│   ├── 05_Spatial_PostProcessing.py
│   ├── 06_GEE_Dam_MNDWI_TimeSeries.py
│   ├── 07_Filter_Roofs_and_Roads.py
│   ├── 08_Sensor_Accuracy_and_Trends.py
│   ├── 09_GEE_Basin_Rainfall.py
│   ├── 10_Structural_Blockage.py
│   ├── 11_Vectorize_MERIT_Rivers.py
│   ├── 12_Delineate_River_Reaches.py
│   ├── 13_GEE_Reach_NDVI_Extraction.py
│   ├── 14_GEE_Catchment_Rainfall.py
│   ├── 15_TWFE_Panel_Analysis.py
│   └── 16_Ceiling_Sensitivity_Tests.py
├── Data/                                 # Baseline GeoPackage & GeoTIFF assets, and script outputs
└── README.md
```

---

## Software and Package Requirements

- **Python Version:** Python `>= 3.9`
- **Required Libraries:**
  ```bash
  pip install earthengine-api geopandas rasterio shapely pandas numpy scipy scikit-learn networkx matplotlib seaborn linearmodels
  ```

---

## Google Earth Engine Authentication

Several scripts interface directly with Google Earth Engine (GEE). Before running GEE-dependent scripts, authenticate your local environment:
```bash
earthengine authenticate
```
In scripts interacting with GEE (Steps 01, 04, 05, 06, 09, 13, and 14), replace the placeholder `GEE_PROJECT = 'YOUR_GEE_PROJECT'` with your active Google Cloud / Earth Engine project ID.  

Note: This workflow utilises datasets and functions native to Google Earth Engine and is subject to change with backend and software updates.

---

## Data Inputs & Directory Configuration

All baseline vector layers are provided in open-standard GeoPackage (`.gpkg`) format, and raster data in compressed GeoTIFF (`.tif`) format, stored in the `Data/` folder.

> [!IMPORTANT]
> **Data Directory Filepath Configuration:**  
> Before running the scripts, set the `DATA_DIR` path variable at the top of each script to the full absolute path of your local `Data/` folder:
> ```python
> DATA_DIR = r"C:\Users\username\Documents\Namibia_Dams\Data"
> ```

### Summary of Baseline Data Assets in `Data/`:

* **Catchment Boundaries:**
  * `Swakop_Basin_Boundary.gpkg`: Polygon boundary of the Swakop River training basin (Step 01).
  * `Ugab_Basin_Boundary.gpkg`: Polygon boundary of the primary Ugab River study catchment (Steps 04, 05, 10, 11).
* **Ground-Truth & Reference Impoundments:**
  * `Swakop_Reference_Dams.gpkg`: Reference active training dams ($n = 1,159$) validated against high-resolution imagery (Step 01).
  * `Swakop_Hard_Negatives.gpkg`: Curated false-positive hard negative polygons ($n = 1,300$) spanning all iterative training rounds (reflective mining pits, wet riverbed sand, urban surfaces) (Step 01).
  * `Ugab_Reference_Dams.gpkg`: Reference ground-truth dams in the Ugab basin used for independent spatial validation (Step 05).
* **Hydrological Networks & Drainage Topography:**
  * `Ugab_HydroRIVERS_Network.gpkg`: HydroRIVERS vector reach segments within the Ugab basin providing network topology (`NEXT_DOWN`) and upland catchment areas (`UPLAND_SKM`) (Step 10).
  * `Ugab_MERIT_Hydro_UPA.tif`: High-resolution (3-arcsecond / ~90 m) MERIT Hydro Upstream Contributing Area raster used to vectorize active channel flowlines (Step 11).
  * `Ugab_Mask_Major_Rivers.gpkg`: MERIT Hydro trunk river channels used to mask out in-channel false positives (Step 05).
* **Ancillary Spatial Exclusion Masks (Ugab Basin):**
  * `Ugab_Mask_Mining.gpkg`: Global mining polygons database subset to exclude open-pit mines and tailings dams (Step 05).
  * `Ugab_Mask_Lakes.gpkg`: HydroLAKES dataset subset to eliminate natural permanent water bodies (Step 05).
  * `Ugab_Mask_Large_Pans.gpkg`: National inventory of large seasonal standing water pans (Step 05).
  * `Ugab_Mask_Small_Pans.gpkg`: National inventory of small springs and seasonal pans (Step 05).
  * `Ugab_Mask_Urban.gpkg`: Natural Earth 10m urban areas mask (Step 05).
  * `Ugab_Mask_OSM_Buildings.gpkg`: OpenStreetMap building footprints used to eliminate reflective metal rooftops (Step 05).
  * `Ugab_Mask_OSM_Urban.gpkg`: OpenStreetMap urban land use polygons (Step 05).
  * `Ugab_Mask_OSM_Roads.gpkg`: OpenStreetMap major transport network buffer used to exclude smooth asphalt and graded roads (Step 05).

---

## Script Summaries

1. **`01_GEE_PointPixel_Extraction.py`**: Extracts 21 multi-sensor optical (Sentinel-2), SAR backscatter (Sentinel-1), and topographical features (NASADEM slope, MERIT HAND) across ground-truth dams ($n = 1,159$), hard negatives from all iterative testing rounds ($n = 1,300$), and dynamically sampled ESA WorldCover background points in the Swakop River basin via Google Earth Engine.
2. **`02_DecisionTree_Optimization.py`**: Trains a depth-constrained, interpretable decision tree classifier on the point-pixel training dataset to identify optimal optical-radar split thresholds for surface water detection.
3. **`03_Topography_Distributions.py`**: Evaluates empirical slope and HAND frequency distributions between true dams and false positives to define upper terrain thresholds that eliminate mountain shadows and upland noise.
4. **`04_GEE_Dam_Detection_Export.py`**: Applies the multi-pathway decision rules and topographical constraints across the entire Ugab River basin in Google Earth Engine and exports candidate water pixels as a GeoTIFF.
5. **`05_Spatial_PostProcessing.py`**: Vectorizes candidate pixels, applies spatial clustering (100 m buffer merge), excludes non-dam water bodies and urban infrastructure via ancillary masks, and verifies terrain attributes via GEE centroid sampling.
6. **`06_GEE_Dam_MNDWI_TimeSeries.py`**: Extracts multi-decadal annual wet-season maximum MNDWI time series for all candidate dams from Sentinel-2 (2016–2026) and Landsat 5/7/8/9 (1990–2026) in Earth Engine.
7. **`07_Filter_Roofs_and_Roads.py`**: Filters candidate polygons using multi-year Sentinel-2 MNDWI dynamics (variance, amplitude, and step-change thresholds) to eliminate smooth, highly reflective anthropogenic surfaces such as metal roofs and asphalt roads.
8. **`08_Sensor_Accuracy_and_Trends.py`**: Evaluates cross-sensor inundation agreement between 30-m Landsat and 10-m Sentinel-2, identifies the historical establishment year for each verified dam, and exports the cleaned Landsat time series.
9. **`09_GEE_Basin_Rainfall.py`**: Extracts the 37-year (1990–2026) annual CHIRPS rainfall time series across the Ugab basin in Earth Engine to contextualize construction tempos against macro-climatic conditions.
10. **`10_Structural_Blockage.py`**: Snaps verified impoundments to the HydroRIVERS drainage network, traverses the directed graph downstream, and calculates cumulative upstream impoundment burden for each river segment.
11. **`11_Vectorize_MERIT_Rivers.py`**: Masks and vectorizes high-resolution MERIT Hydro flowlines (upstream contributing area $> 10\text{ km}^2$) to accurately delineate active braided ephemeral river channels.
12. **`12_Delineate_River_Reaches.py`**: Delineates 414 sequential downstream evaluation reaches (Near, Mid, Far) below dams and samples 100 geographically matched pristine reference reaches, fusing corridors with MERIT Hydro flowlines.
13. **`13_GEE_Reach_NDVI_Extraction.py`**: Extracts 36-year (1990–2025) annual dry-season vegetation metrics (NDVI, NDMI), post-flood green flush, and water frequencies across all 514 river reaches in Earth Engine.
14. **`14_GEE_Catchment_Rainfall.py`**: Extracts annual wet-season and 90-day antecedent precipitation across HydroBASINS Level-12 sub-catchments in Earth Engine for reach-level contributing area weighting.
15. **`15_TWFE_Panel_Analysis.py`**: Estimates the Two-Way Fixed Effects panel econometric models with standard errors clustered across 61 independent hydrological tributary sub-basins to quantify the attenuation of riparian vegetation responsiveness to rainfall.
16. **`16_Ceiling_Sensitivity_Tests.py`**: Performs ceiling effect sensitivity tests—including baseline-rainfall interaction modeling and stratification across baseline greenness tertiles—to confirm that dampening effects are driven by upstream hydrologic blockage rather than an ecological vegetation ceiling.
