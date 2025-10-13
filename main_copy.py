from helpers_new import *
import os
import glob

### inputs 
section_lines_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/Section line/new_section_lines.shp"

## Dtms
dtm_itr1_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/DEM_itr_1.tif"
dtm_itr1_year_ip =  2024                              
dtm_itr1_year_ip = dtm_itr1_year_ip if dtm_itr1_year_ip else "itr1"

dtm_itr2_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/DEM_itr_2.tif"
dtm_itr2_year_ip = 2025                              
dtm_itr2_year_ip = dtm_itr2_year_ip if dtm_itr2_year_ip else "itr2"

output_dir = "D:/2_Analytics/6_plan_vs_actual/8_sept_outputs_3"

def get_linear_outputs(output_dir):
    shape_folder = os.path.join(output_dir, "shape_file_outputs")

    base_names = [
        "planned_and_done_excavation",
        "planned_and_not_done_excavation",
        "unplanned_and_done_excavation",
        "planned_and_done_dump",
        "planned_and_not_done_dump",
        "unplanned_and_done_dump",
    ]

    results = []
    for base in base_names:
        pattern = os.path.join(shape_folder, f"{base}.*")
        matches = glob.glob(pattern)
        results.append(matches[0] if matches else None)

    return tuple(results)

(done_exv,
not_done_exv,
unplanned_exv,
done_dump,
not_done_dump,
unplanned_dump,
) = get_linear_outputs(output_dir)

print("Planned and Done Excavation:", done_exv)
print("Planned and Not Done Excavation:", not_done_exv)
print("Unplanned and Done Excavation:", unplanned_exv)
print("Planned and Done Dump:", done_dump)
print("Planned and Not Done Dump:", not_done_dump)
print("Unplanned and Done Dump:", unplanned_dump)

## boundary polygons
planned_and_done_excavation_path = done_exv
unplanned_and_done_excavtion_path = unplanned_exv
planned_and_used_dump_path = done_dump
unplanned_and_used_dump_path = unplanned_dump


## output folder path 
output_folder_path = "D:/2_Analytics/6_plan_vs_actual/macrel_data/3d_output"

# Load all section lines
section_lines_gdf = gpd.read_file(section_lines_path)
print(f"Total section lines: {len(section_lines_gdf)}")

def process_lines(section_lines_gdf, start_line=1, end_line=None):
    
    # Set end_line to the total number of rows if not specified
    if end_line is None:
        end_line = len(section_lines_gdf)

    all_section_results = []  # Collect data for final CSV here

    # Loop through the specified range
    for i, (idx, line_gdf) in enumerate(
        section_lines_gdf.iloc[start_line - 1:end_line].iterrows(),
        start=start_line
    ):
        print(f"\nProcessing line {i}/{end_line}")
        

        # Convert single row to GeoDataFrame
        line_gdf = gpd.GeoDataFrame([line_gdf], crs=section_lines_gdf.crs)
        print(len(line_gdf), "features in line_gdf")
        if len(line_gdf) != 1:
            print(f"Warning: line_gdf contains {len(line_gdf)} features instead of 1.")

        # Extract section name
        section_value = None
        for attr_name in ['SECTION', 'section']:
            if attr_name in line_gdf.columns:
                section_value = str(line_gdf.iloc[0][attr_name])
                break
    
        # Sanitize and create subfolder
        if section_value:
            section_name = section_value.replace(" ", "_").replace("/", "_").replace("\\", "_")
            section_folder = os.path.join(output_folder_path, f"section_{section_name}")
        else:
            section_name = f"line_{i+1}"
            section_folder = os.path.join(output_folder_path, section_name)
        #print(f"Section folder: {section_folder}")
        os.makedirs(section_folder, exist_ok=True)

        # Intersect clipped line with excavation/dump polygons
        (lines_planned_and_done_excavation,
            lines_unplanned_and_done_excavation,
            lines_planned_and_used_dump,
            lines_unplanned_and_used_dump
        ), section_name =               intersect_line_with_polygons_return_separately(
                                        line_gdf=line_gdf,
                                        output_folder_path=section_folder,
                                        section_name=section_name,
                                        planned_and_done_excavation_path=planned_and_done_excavation_path,
                                        unplanned_and_done_excavation_path=unplanned_and_done_excavtion_path,
                                        planned_and_used_dump_path=planned_and_used_dump_path,
                                        unplanned_and_used_dump_path=unplanned_and_used_dump_path
                                        )       

        print(f"Section name: {section_name}")

        # Extract elevation data from each DTM
        planned_and_done_excavation = extract_segment_df(lines_planned_and_done_excavation, line_gdf, dtm_itr1_path, dtm_itr2_path)
        unplanned_and_done_excavation = extract_segment_df(lines_unplanned_and_done_excavation, line_gdf, dtm_itr1_path, dtm_itr2_path)
        planned_and_used_dump = extract_segment_df(lines_planned_and_used_dump, line_gdf, dtm_itr1_path, dtm_itr2_path)
        unplanned_and_used_dump = extract_segment_df(lines_unplanned_and_used_dump, line_gdf, dtm_itr1_path, dtm_itr2_path)

        segment_data_dict = {
            "planned_and_done_excavation": (planned_and_done_excavation, "green"),
            "unplanned_and_done_excavation": (unplanned_and_done_excavation, "red"),
            "planned_and_used_dump": (planned_and_used_dump, "green"),
            "unplanned_and_used_dump": (unplanned_and_used_dump, "red"),
        }

        # Get elevation profiles
        elevations_itr1 = get_elevation_points_from_lines(dtm_itr1_path, line_gdf, interval=0.01)
        elevations_itr2 = get_elevation_points_from_lines(dtm_itr2_path, line_gdf, interval=0.01)

        # Convert elevation to DataFrame
        df1 = pd.DataFrame(elevations_itr1, columns=["x", "y", "z"])
        df1["chainage"] = [0.01 * i for i in range(len(df1))]

        df2 = pd.DataFrame(elevations_itr2, columns=["x", "y", "z"])
        df2["chainage"] = [0.01 * i for i in range(len(df2))]

        # Plot and save
        plot_multiple_intersections_on_elevation(
            df1, dtm_itr1_year_ip, 
            df2, dtm_itr2_year_ip, 
            segment_data_dict,
            section_name=section_name,
            output_folder_path=section_folder
        )

        # Store result for this section
        all_section_results.append((section_name, segment_data_dict))

        
        output_dxf_path = os.path.join(section_folder, f"{section_name}_elevation_profile.dxf")
        export_to_dxf_with_colors_updated(df1, df2, segment_data_dict, output_dxf_path, section_name)

    return all_section_results
        

all_section_results = process_lines(section_lines_gdf, start_line=1, end_line=None)

print(all_section_results)

# Export one CSV file
export_all_section_deviation_summary_csv(all_section_results, output_folder_path)



