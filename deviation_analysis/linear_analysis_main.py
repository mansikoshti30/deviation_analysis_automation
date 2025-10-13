import geopandas as gpd
from tkinter import filedialog,messagebox
import tkinter as tk
import pandas as pd
import sys,os

from helpers_linear_analysis import *
from get_section_lines import *
from helpers_sectional_analysis import *


## output folder path
output_dir = "D:/2_Analytics/6_plan_vs_actual/8_sept_outputs_4"
os.makedirs(output_dir, exist_ok=True)

## actual data from site (excavation and dump)
actual_excavation_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Excavated area.shp"
actual_dump_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Dump area.shp"

## planned/proposed data from client 
planned_excavtaion_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed PIT AREA from client.shp"
planned_dump_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed Dump from client.shp"


## dwg file and ODA convertor
input_folder_dwg_file = "D:/2_Analytics/6_plan_vs_actual/raw_data_dwg_file/dwg_file"
oda_exe = "C:/Program Files/ODA/ODAFileConverter 26.7.0/ODAFileConverter.exe"
section_line_folder = os.path.join(output_dir, "section_lines")
os.makedirs(section_line_folder, exist_ok=True)



### sectional inputs 
## Dtms
dtm_itr1_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/DEM_itr_1.tif"
dtm_itr1_year_ip =  2024                              
dtm_itr1_year_ip = dtm_itr1_year_ip if dtm_itr1_year_ip else "itr1"

dtm_itr2_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/DEM_itr_2.tif"
dtm_itr2_year_ip = 2025                              
dtm_itr2_year_ip = dtm_itr2_year_ip if dtm_itr2_year_ip else "itr2"


## preprocess the data 
actual_excavation, actual_dump, planned_excavation, planned_dump, epsg_code =  get_data_preprocessed(
                                                                                        act_exv = actual_excavation_inputs, 
                                                                                        act_dump = actual_dump_inputs, 
                                                                                        pln_exv= planned_excavtaion_inputs, 
                                                                                        pln_dump= planned_dump_inputs, 
                                                                                        output_dir= output_dir
                                                                                        )

print(epsg_code)

## save results in csv file 
results_exv, df_exv = split_planned_done_new(done_file= actual_excavation, 
                       planned_file= planned_excavation, 
                       output_dir= output_dir, 
                       area_type= "excavation" )


results_dump, df_dump = split_planned_done_new(done_file= actual_dump, 
                       planned_file= planned_dump, 
                       output_dir= output_dir, 
                       area_type= "dump") 


## merge the csv
excel_path = merge_and_save(df_exv, df_dump, output_dir)

# Save the pie chart and get df to append in final excel sheet 
df_excavation, img_path_excavation = plot_category_chart(output_dir, category="excavation", labels=None, colors=None, figsize=(4,4))
df_dump, img_path_dump = plot_category_chart(output_dir, category="dump", labels=None, colors=None, figsize=(4,4))

update_excel_with_tables_charts(
    input_excel = excel_path, 
    df_excavation = df_excavation, 
    img_path_excavation = img_path_excavation,
    df_dump=df_dump, 
    img_path_dump= img_path_dump
)



section_lines_shp = get_section_lines(input_folder= input_folder_dwg_file,
                                         section_line_folder= section_line_folder,
                                         oda_exe=oda_exe,
                                         crs = epsg_code)



## 3d analysis 
(done_exv, not_done_exv, unplanned_exv, done_dump, not_done_dump, unplanned_dump) = get_linear_outputs(output_dir)

# create the separate folder for sectional analysis 
sectional_analysis_output_path = os.path.join(output_dir,"sectional_output")
os.makedirs(sectional_analysis_output_path, exist_ok=True)

# Load all section lines
section_lines_gdf = gpd.read_file(section_lines_shp)
print(f"Total section lines: {len(section_lines_gdf)}")

## read the 1st line in section line 

section_lines_gdf = gpd.read_file(section_lines_shp)


## get 1st line 
first_feature = section_lines_gdf.iloc[1]
print(first_feature)
start_text = first_feature["start_text"]
end_text = first_feature["end_text"]

print("Start text:", start_text)
print("End text:", end_text)