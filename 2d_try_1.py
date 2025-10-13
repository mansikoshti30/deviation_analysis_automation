import geopandas as gpd
from tkinter import filedialog,messagebox
import tkinter as tk
import pandas as pd
import sys,os

from helpers_2d_new import *


#from helpers_2d import *


## output folder path
output_dir = "D:/2_Analytics/6_plan_vs_actual/1_sept_outputs_11"
os.makedirs(output_dir, exist_ok=True)


## actual data from site (excavation and dump)
actual_excavation_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Excavated area.shp"
actual_dump_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Dump area.shp"

## planned/proposed data from client 
planned_excavtaion_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed PIT AREA from client.shp"
planned_dump_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed Dump from client.shp"


## preprocess the data 
actual_excavation, actual_dump, planned_excavation, planned_dump =  get_data_preprocessed(
                                                                                        act_exv = actual_excavation_inputs, 
                                                                                        act_dump = actual_dump_inputs, 
                                                                                        pln_exv= planned_excavtaion_inputs, 
                                                                                        pln_dump= planned_dump_inputs, 
                                                                                        output_dir= output_dir
                                                                                        )

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

print(df_excavation)
print()
print(df_dump)


update_excel_with_tables_charts(
    input_excel = excel_path, 
    output_excel="D:/2_Analytics/6_plan_vs_actual/1_sept_outputs_11/updated_merged_summary.xlsx",
    df_excavation = df_excavation, 
    img_path_excavation = img_path_excavation,
    df_dump=df_dump, 
    img_path_dump= img_path_dump
)