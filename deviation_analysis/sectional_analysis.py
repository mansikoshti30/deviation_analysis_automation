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




## read the 1st line in section line 

section_lines_gdf = gpd.read_file("D:/2_Analytics/6_plan_vs_actual/8_sept_outputs_4/section_lines/LINES_with_start_end_text.shp")


## get 1st line 
first_line_gdf = section_lines_gdf.iloc[[0]]
start_text = first_line_gdf["start_text"]
end_text = first_line_gdf["end_text"]

print(start_text, end_text)



## get output from linear analysis 
(done_exv, not_done_exv, unplanned_exv, done_dump, not_done_dump, unplanned_dump) = get_linear_outputs(output_dir)
print(done_exv)

sectional_analysis_output_path = os.path.join(output_dir,"sectional_output")
os.makedirs(sectional_analysis_output_path, exist_ok=True)


# Intersect clipped line with excavation/dump polygons
(lines_planned_and_done_excavation,
lines_unplanned_and_done_excavation,
lines_planned_and_used_dump,
lines_unplanned_and_used_dump), section_name =intersect_line_with_polygons_return_separately(
                                                line_gdf=first_line_gdf,
                                                output_folder_path= sectional_analysis_output_path,
                                                section_name= "J J'",
                                                planned_and_done_excavation_path = done_exv,
                                                unplanned_and_done_excavation_path = unplanned_exv,
                                                planned_and_used_dump_path = done_dump,
                                                unplanned_and_used_dump_path  = unplanned_dump)       