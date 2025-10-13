import glob
import sys,os
import geopandas as gpd
import numpy as np
from tkinter import filedialog,messagebox
import tkinter as tk
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple
import re
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point




def get_linear_outputs(output_dir):
    shape_folder = os.path.join(output_dir, "shapefile_outputs")

    base_names = [
        "planned_and_done_excavation",
        "unplanned_and_done_excavation",
        "planned_and_done_dump",
        "unplanned_and_done_dump",
    ]
    results = []
    for base in base_names:
        pattern = os.path.join(shape_folder, f"{base}.shp")  # only .shp
        matches = glob.glob(pattern)
        results.append(matches[0] if matches else None)

    return tuple(results)


def elevation_at_point(dtm_src, x, y):
    """Query elevation value from an already opened DTM raster at (x, y)."""
    q_ele = list(dtm_src.sample([(x, y)]))[0][0]
    return q_ele

# --- step 2: main combined sampler ---
def sample_elevations_two_dtms(
    dtm_path1: str,
    dtm_path2: str,
    section_gdf: gpd.GeoDataFrame,
    section_number,
    interval: float = 0.01
) -> pd.DataFrame:
    """
    Interpolates points along the *first* feature in section_gdf,
    fetches elevations from both DTMs, and returns a DataFrame.

    Columns: ['chainage','x','y','z_itr1','z_itr2']
    """
    # Ensure we only take first feature as GeoDataFrame
    
    line_gdf = section_gdf.iloc[[section_number]]

    # Get the LineString geometry
    geom = line_gdf.geometry.iloc[0]
    if not isinstance(geom, LineString):
        raise ValueError("First feature is not a LineString!")

    length = geom.length
    num_points = int(length // interval) + 1
    distances = [i * interval for i in range(num_points + 1)]
    if distances[-1] < length:  # make sure last point is included
        distances.append(length)

    rows = []
    with rasterio.open(dtm_path1) as dtm1, rasterio.open(dtm_path2) as dtm2:
        # Reproject line_gdf if needed
        if line_gdf.crs is not None:
            if line_gdf.crs != dtm1.crs:
                print("Reprojecting section line to match DTM CRS...")
                line_gdf = line_gdf.to_crs(dtm1.crs)
                geom = line_gdf.geometry.iloc[0]

        for d in distances:
            pt = geom.interpolate(d)
            x, y = pt.x, pt.y
            z1 = elevation_at_point(dtm1, x, y)
            z2 = elevation_at_point(dtm2, x, y)
            rows.append((d, x, y, z1, z2))

    df = pd.DataFrame(rows, columns=["chainage", "x", "y", "z_itr1", "z_itr2"])
    return df


def intersect_line_with_polygons_return_separately(line_gdf, 
                                                   output_folder_path: str,
                                                   planned_and_done_excavation_path: str = None,
                                                   unplanned_and_done_excavation_path: str = None,
                                                   planned_and_used_dump_path: str = None,
                                                   unplanned_and_used_dump_path: str = None):
    def process(name, poly_path, section_output_path):
        output_path = os.path.join(section_output_path, f"section_{section_name}_{name}.shp")
        print(output_path)

        # Return empty GeoDataFrame if path is None or file does not exist
        if poly_path is None or not os.path.exists(poly_path):
            print(f"{name}: File path is None or does not exist.")
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs=line_gdf.crs)
            empty_gdf.to_file(output_path)
            return empty_gdf

        poly_gdf = gpd.read_file(poly_path)
        if poly_gdf.empty:
            print(f"{name}: Polygon file is empty.")
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs=line_gdf.crs)
            empty_gdf.to_file(output_path)
            return empty_gdf

        # Reproject to match CRS
        if poly_gdf.crs != line_gdf.crs:
            poly_gdf = poly_gdf.to_crs(line_gdf.crs)

        poly_union = poly_gdf.union_all()
        intersection = line_geom.intersection(poly_union)

        segments = []
        if intersection.is_empty:
            print(f"{name}: No intersection.")
        elif isinstance(intersection, LineString):
            segments.append(intersection)
        elif isinstance(intersection, MultiLineString):
            segments.extend(intersection.geoms)
        else:
            print(f"{name}: Unsupported geometry type {type(intersection)}")

        # Prepare GeoDataFrame
        intersect_gdf = gpd.GeoDataFrame(geometry=segments, crs=poly_gdf.crs)

        if not intersect_gdf.empty:
            intersect_gdf["temp_id"] = intersect_gdf.index
            intersect_gdf = gpd.sjoin(intersect_gdf, poly_gdf[["geometry", "AREA_NAME"]], how="left", predicate="intersects")
            intersect_gdf = intersect_gdf.drop(columns=["index_right", "temp_id"])

        # Save output
        intersect_gdf.to_file(output_path)
        return intersect_gdf

    # Ensure only one line feature is present
    if len(line_gdf) != 1:
        raise ValueError("Line GeoDataFrame must contain exactly one feature.")
    line_geom = line_gdf.geometry.iloc[0]

    row = line_gdf.iloc[0]                       # first row
    section_name = f"{row['start_text']}_{row['end_text']}"
    print(section_name)

    

    # Process each layer
    lines_planned_and_done_excavation = process("planned_and_done_excavation", planned_and_done_excavation_path, output_folder_path)
    lines_unplanned_and_done_excavation = process("unplanned_and_done_excavation", unplanned_and_done_excavation_path, output_folder_path)
    lines_planned_and_used_dump = process("planned_and_used_dump", planned_and_used_dump_path, output_folder_path)
    lines_unplanned_and_used_dump = process("unplanned_and_used_dump", unplanned_and_used_dump_path, output_folder_path)

    return (
        lines_planned_and_done_excavation,
        lines_unplanned_and_done_excavation,
        lines_planned_and_used_dump,
        lines_unplanned_and_used_dump
    ), section_name



def add_labels_for_planned_area(
    section_df: pd.DataFrame,
    line_gdf,
    key: str,
    output_csv: str = None
):
    # Ensure required columns exist
    if "line_name" not in section_df.columns:
        section_df["line_name"] = np.nan
    if "area_name" not in section_df.columns:
        section_df["area_name"] = pd.Series([None] * len(section_df), dtype="object")
    # NEW: ensure label_name column exists
    if "label_name" not in section_df.columns:
        section_df["label_name"] = pd.Series([None] * len(section_df), dtype="object")

    # Early exit if no line_gdf provided
    if line_gdf is None:
        print(f"Skipped: line_gdf is None for key '{key}'.")
    elif line_gdf.empty:
        print(f"Skipped: line_gdf has no features for key '{key}'.")
    else:
        if "AREA_NAME" not in line_gdf.columns:
            raise ValueError("line_gdf must contain an 'AREA_NAME' column.")

        # Build the full section line
        section_line = LineString(section_df[['x','y']].to_numpy())
        chainages = section_df['chainage'].to_numpy()

        for idx, geom in line_gdf.iterrows():
            line = geom.geometry
            area_name = geom["AREA_NAME"] if "AREA_NAME" in geom else None

            if not isinstance(line, LineString):
                continue  # skip non-LineString

            # Start & end points
            start_pt = Point(line.coords[0])
            end_pt = Point(line.coords[-1])

            # Project onto section line
            start_chain_exact = section_line.project(start_pt)
            end_chain_exact = section_line.project(end_pt)

            # Snap to nearest sampled chainages
            start_chain = chainages[np.argmin(np.abs(chainages - start_chain_exact))]
            end_chain = chainages[np.argmin(np.abs(chainages - end_chain_exact))]

            # Ensure start <= end
            start_chain, end_chain = sorted([start_chain, end_chain])

            # Mask rows in section_df within this range and set labels
            mask = (section_df["chainage"] >= start_chain) & (section_df["chainage"] <= end_chain)
            section_df.loc[mask, "line_name"] = key
            section_df.loc[mask, "area_name"] = area_name

            # Derive label_name from area_name (Excavation Area i / Dump Area i)
            label_value = None
            if isinstance(area_name, str):
                lower = area_name.lower()
                kind = None
                if "excavation" in lower:
                    kind = "Excavation Area"
                elif "dump" in lower:
                    kind = "Dump Area"

                # find the last integer in the string (e.g., area_23 -> 23)
                m = re.search(r'(\d+)(?!.*\d)', area_name)
                if kind and m:
                    label_value = f"{kind} {m.group(1)}"
                elif kind:
                    # if kind found but no number, use kind only
                    label_value = kind
                else:
                    # fallback: None (or you could set to area_name if you prefer)
                    label_value = None

            # apply label_value to the same mask rows
            section_df.loc[mask, "label_name"] = label_value

    # Save to CSV if requested
    if output_csv:
        section_df.to_csv(output_csv, index=False)
        print(f"Updated DataFrame exported to: {output_csv}")

    return section_df



def add_labels_for_unplanned_area(
    section_df: pd.DataFrame,
    line_gdf,
    key: str,
    threshold: float,
    output_csv: str = None
):
    """
    Label unplanned areas only if max abs(z_itr1 - z_itr2) in the area exceeds `threshold`.
    Also create/populate `label_name` like "Excavation Area <i>" or "Dump Area <i>" (case-sensitive).
    """

    # Ensure required elevation columns exist
    required = {"chainage", "x", "y", "z_itr1", "z_itr2"}
    if not required.issubset(section_df.columns):
        missing = required - set(section_df.columns)
        raise ValueError(f"section_df missing required columns: {missing}")

    # Ensure line_name, area_name, label_name columns exist
    if "line_name" not in section_df.columns:
        section_df["line_name"] = np.nan
    if "area_name" not in section_df.columns:
        section_df["area_name"] = pd.Series([None] * len(section_df), dtype="object")
    if "label_name" not in section_df.columns:
        section_df["label_name"] = pd.Series([None] * len(section_df), dtype="object")

    # Early exit if no line_gdf provided
    if line_gdf is None:
        print(f"Skipped: line_gdf is None for key '{key}'.")
    elif getattr(line_gdf, "empty", False):
        print(f"Skipped: line_gdf has no features for key '{key}'.")
    else:
        has_area_name = "AREA_NAME" in line_gdf.columns

        # Build the full section line from the sampled coordinates
        section_line = LineString(section_df[['x', 'y']].to_numpy())
        chainages = section_df['chainage'].to_numpy()

        # helper to derive clean label_name from raw area_name
        def derive_label(area_name: str) -> str:
            if not isinstance(area_name, str):
                return None
            lower = area_name.lower()
            kind = None
            if "excavation" in lower:
                kind = "Excavation Area"
            elif "dump" in lower:
                kind = "Dump Area"
            # find the last integer in the string (if any)
            m = re.search(r'(\d+)(?!.*\d)', area_name)
            if kind and m:
                return f"{kind} {m.group(1)}"
            elif kind:
                return kind
            return None

        for idx, row in line_gdf.iterrows():
            line = row.geometry
            if not isinstance(line, LineString):
                # skip non-LineString features
                continue

            area_name = row["AREA_NAME"] if has_area_name else None

            # Start & end points of the blue line
            start_pt = Point(line.coords[0])
            end_pt = Point(line.coords[-1])

            # Project onto section line -> exact continuous chainage
            start_chain_exact = section_line.project(start_pt)
            end_chain_exact = section_line.project(end_pt)

            # Snap to nearest sampled chainages in section_df
            start_chain = chainages[np.argmin(np.abs(chainages - start_chain_exact))]
            end_chain = chainages[np.argmin(np.abs(chainages - end_chain_exact))]

            # Ensure start <= end
            start_chain, end_chain = sorted([start_chain, end_chain])

            # Mask rows in section_df within this range
            mask = (section_df["chainage"] >= start_chain) & (section_df["chainage"] <= end_chain)

            # Compute absolute diffs for this range
            diffs = (section_df.loc[mask, "z_itr1"] - section_df.loc[mask, "z_itr2"]).abs()

            if (diffs > threshold).any():
                # Significant deviation: keep the label and set area_name & label_name
                section_df.loc[mask, "line_name"] = key
                section_df.loc[mask, "area_name"] = area_name
                section_df.loc[mask, "label_name"] = derive_label(area_name)
            else:
                # No significant deviation: clear any label/area_name/label_name for this range
                section_df.loc[mask, "line_name"] = np.nan
                section_df.loc[mask, "area_name"] = np.nan
                section_df.loc[mask, "label_name"] = np.nan
                print(f"Removed label for '{key}' at feature {idx}: no elevation diff > {threshold}")

    # Save to CSV if requested
    if output_csv:
        section_df.to_csv(output_csv, index=False)
        print(f"Updated DataFrame exported to: {output_csv}")

    return section_df