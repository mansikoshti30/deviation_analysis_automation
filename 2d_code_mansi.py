import geopandas as gpd
import pandas as pd
import os


def read_and_clean(path):
    """Read shapefile and remove duplicate geometries"""
    gdf = gpd.read_file(path)
    return gdf.drop_duplicates(subset="geometry").reset_index(drop=True)

def add_area_column(gdf):
    """Explode multipolygons and add area in hectares"""
    gdf = gdf.explode().reset_index(drop=True)
    gdf["Area_Name"] = "Area " + (gdf.index + 1).astype(str)
    gdf["Area_Ha"] = gdf.area / 10000
    return gdf[["Area_Name", "Area_Ha", "geometry"]]

def classify_actual_vs_planned(actual_gdf, planned_gdf, mine_boundary_gdf, label_prefix):
    """Classify actual areas into categories"""
    results = {}

    for i, row in actual_gdf.iterrows():
        area_gdf = gpd.GeoDataFrame([row], crs=actual_gdf.crs)

        # Outside lease
        outside = gpd.overlay(area_gdf, mine_boundary_gdf, how="difference")
        results.setdefault("Outside", []).append(outside)

        # Planned & Done
        planned_done = gpd.overlay(area_gdf, planned_gdf, how="intersection")
        results.setdefault("Planned", []).append(planned_done)

        # Unplanned & Done
        unplanned = gpd.overlay(area_gdf, planned_gdf, how="difference")
        results.setdefault("Unplanned", []).append(unplanned)

    # Merge lists
    for key in results:
        results[key] = gpd.GeoDataFrame(pd.concat(results[key], ignore_index=True))

        # Add area column
        if not results[key].empty:
            results[key]["Area_Ha"] = results[key].area / 10000
            results[key]["Section_Name"] = (
                {"Planned": "P", "Unplanned": "U", "Outside": "O"}[key]
                + label_prefix
                + "_"
                + (results[key].index + 1).astype(str)
            )

    # Planned but not done
    not_done = gpd.overlay(planned_gdf, actual_gdf, how="difference")
    if not not_done.empty:
        not_done["Area_Ha"] = not_done.area / 10000
        not_done["Section_Name"] = "ND_" + label_prefix
        results["NotDone"] = not_done

    return results

def save_outputs(results, output_dir, name):
    """Save shapefiles, geojsons and return dataframe summary"""
    os.makedirs(output_dir, exist_ok=True)
    df_list = []

    for key, gdf in results.items():
        if not gdf.empty:
            out_shp = os.path.join(output_dir, f"{name}_{key}.shp")
            out_json = os.path.join(output_dir, f"{name}_{key}.geojson")
            gdf.to_file(out_shp, index=False)
            gdf.to_file(out_json, index=False)

            df = gdf[["Area_Ha"]].copy()
            df["Category"] = key
            df_list.append(df.groupby("Category").sum())

    return pd.concat(df_list) if df_list else pd.DataFrame()




## workflow 

# Inputs
output_dir = "D:/2_Analytics/6_plan_vs_actual/demo_test_2d_output_2"

mine_boundary_path = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Mine Boundary/Mine Boundary.shp"

actual_excav_path = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Excavated area.shp"
planned_excav_path = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed PIT AREA from client.shp"

actual_dump_path = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Dump area.shp"
planned_dump_path = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed Dump from client.shp"

# load the data
mine_boundary = read_and_clean(mine_boundary_path)
actual_excav = add_area_column(read_and_clean(actual_excav_path))
planned_excav = add_area_column(read_and_clean(planned_excav_path))
actual_dump = add_area_column(read_and_clean(actual_dump_path))
planned_dump = add_area_column(read_and_clean(planned_dump_path))

# Classify
excav_results = classify_actual_vs_planned(actual_excav, planned_excav, mine_boundary, "EA")
dump_results = classify_actual_vs_planned(actual_dump, planned_dump, mine_boundary, "DA")

# Save & Summarize
excav_summary = save_outputs(excav_results, os.path.join(output_dir, "Excavation"), "Excavation")
dump_summary = save_outputs(dump_results, os.path.join(output_dir, "Dump"), "Dump")

# Final CSV
summary = pd.concat([excav_summary, dump_summary])
summary.to_csv(os.path.join(output_dir, "Excavation_and_Dump_Analysis.csv"))
