import geopandas as gpd
from tkinter import filedialog,messagebox
import tkinter as tk
import pandas as pd
import sys,os

from helpers_2d import *

#Inputs

output_dir = "D:/2_Analytics/6_plan_vs_actual/demo_test_2d_output"

mine_boundary = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Mine Boundary/Mine Boundary.shp"

actual_excavation_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Excavated area.shp"
planned_excavtaion_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed PIT AREA from client.shp"

actual_dump_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Dump area.shp"
planned_dump_inputs = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed Dump from client.shp"





actual_excavation_gdf = delete_duplicates(actual_excavation_inputs)
planned_excavation_gdf = delete_duplicates(planned_excavtaion_inputs)

mine_boundary_gdf = delete_duplicates(mine_boundary)

actual_dump_gdf = delete_duplicates(actual_dump_inputs)
planned_dump_gdf = delete_duplicates(planned_dump_inputs)




actual_excavation_gdf = actual_area_gdf_explode_and_add_area_column(actual_excavation_gdf)
actual_dump_gdf =  actual_area_gdf_explode_and_add_area_column(actual_dump_gdf)

os.makedirs(fr'{output_dir}\Actual_Dump_and_Excavation_with_Name',exist_ok=True)
actual_excavation_gdf.to_file(fr'{output_dir}\Actual_Dump_and_Excavation_with_Name\Actual_Excavation_with_Area_name.shp',index=False)
actual_dump_gdf.to_file(fr'{output_dir}\Actual_Dump_and_Excavation_with_Name\Actual_Dump_with_Area_name.shp',index=False)


number_of_dumps = len(actual_dump_gdf)
number_of_excavation_areas = len(actual_excavation_gdf)


outside_lease_boundary_excavation_gdfs_list, planned_and_done_excavation_gdfs_list,unplanned_and_done_excavation_gdfs_list = create_gdf_from_each_feature(actual_excavation_gdf, planned_excavation_gdf)


outside_lease_boundary_excavation_gdf = merge_geodataframe(outside_lease_boundary_excavation_gdfs_list,'Pit','Outside')
planned_and_done_excavation_gdf = merge_geodataframe(planned_and_done_excavation_gdfs_list,'Pit','Planned')
unplanned_and_done_excavation_gdf = merge_geodataframe(unplanned_and_done_excavation_gdfs_list,'Pit','Unplanned')

outside_lease_boundary_dump_gdfs_list, planned_and_used_dump_gdfs_list, unplanned_and_used_dump_gdfs_list = create_gdf_from_each_feature(actual_dump_gdf,planned_dump_gdf) 

outside_lease_boundary_dump_area_gdf = merge_geodataframe(outside_lease_boundary_dump_gdfs_list,'Dump','Outside')
planned_and_used_dump_area_gdf = merge_geodataframe(planned_and_used_dump_gdfs_list,'Dump','Planned')
unplanned_and_used_dump_area_gdf = merge_geodataframe(unplanned_and_used_dump_gdfs_list,'Dump','Unplanned')


planned_and_not_done_excavation_gdf = gpd.overlay(planned_excavation_gdf,actual_excavation_gdf,how='difference')
planned_and_not_done_excavation_gdf = planned_and_not_done_excavation_gdf.dissolve()
planned_and_not_done_excavation_gdf['Area_Name'] = 'Area ' + str(1)
planned_and_not_done_excavation_gdf['Area_Ha'] = (planned_and_not_done_excavation_gdf.area/10000).round(2)

planned_and_not_used_dump_gdf = gpd.overlay(planned_dump_gdf,actual_dump_gdf,how='difference')
planned_and_not_used_dump_gdf = planned_and_not_used_dump_gdf.dissolve()
planned_and_not_used_dump_gdf['Area_Name'] = 'Area' + str(1)
planned_and_not_used_dump_gdf['Area_Ha'] = (planned_and_not_used_dump_gdf.area/10000).round(2)





outside_lease_boundary_excavation_df = create_shapefiles(outside_lease_boundary_excavation_gdf,'Outside_Lease_Boundary_Excavation',number_of_excavation_areas,'Pit','Outside')
planned_and_done_excavation_df = create_shapefiles(planned_and_done_excavation_gdf,'Planned_and_Done_Excavation',number_of_excavation_areas,'Pit','Planned')
unplanned_and_done_excavation_df = create_shapefiles(unplanned_and_done_excavation_gdf,'Unplanned_and_Done_Excavation',number_of_excavation_areas,'Pit','Unplanned')
outside_lease_boundary_dump_area_df = create_shapefiles(outside_lease_boundary_dump_area_gdf,'Outside_Lease_Boundary_Dump_Area',number_of_dumps,'Dump','Outside')
planned_and_used_dump_area_df= create_shapefiles(planned_and_used_dump_area_gdf,'Planned_and_Used_Dump_Area',number_of_dumps,'Dump','Planned')
unplanned_and_used_dump_area_df= create_shapefiles(unplanned_and_used_dump_area_gdf,'Unplanned_and_Used_Dump_Area',number_of_dumps,'Dump','Unplanned')

planned_and_not_done_excavation_df = create_shapefiles(planned_and_not_done_excavation_gdf,'Planned_and_Not_Done_Excavation',number_of_excavation_areas,'Pit','Not_Used')
planned_and_not_used_dump_df = create_shapefiles(planned_and_not_used_dump_gdf,'Planned_and_Not_Used_Dump_Area',number_of_dumps,'Dump','Not_Used')




    

outside_lease_boundary_excavation_df = correct_area_name(outside_lease_boundary_excavation_df,'Unplanned and active - Critical (in Ha)')
planned_and_done_excavation_df = correct_area_name(planned_and_done_excavation_df,'Planned and Active (in Ha)')
unplanned_and_done_excavation_df = correct_area_name(unplanned_and_done_excavation_df,'Unplanned And Active (in Ha)')

outside_lease_boundary_dump_area_df = correct_area_name(outside_lease_boundary_dump_area_df,'Unplanned and active - Critical (in Ha)')
planned_and_used_dump_area_df = correct_area_name(planned_and_done_excavation_df,'Planned and Active (in Ha)')
unplanned_and_used_dump_area_df = correct_area_name(unplanned_and_used_dump_area_df,'Unplanned And Active (in Ha)')


planned_and_not_excavated_area = planned_and_not_done_excavation_gdf['Area_Ha'].iloc[0]
planned_and_not_used_dump_area =  planned_and_not_used_dump_gdf['Area_Ha'].iloc[0]




excavation_analysis_df = merge_dataframes(outside_lease_boundary_excavation_df,planned_and_done_excavation_df,unplanned_and_done_excavation_df)

dump_area_analysis_df = merge_dataframes(outside_lease_boundary_dump_area_df,planned_and_used_dump_area_df,unplanned_and_used_dump_area_df)

excavation_analysis_df = excavation_analysis_df.fillna(0)
dump_area_analysis_df = dump_area_analysis_df.fillna(0)

excavation_analysis_df['Total Area Excavation Planned and Not Done (in Ha)'] = planned_and_not_excavated_area
dump_area_analysis_df['Total Area Dump Planned and Inactive (in Ha)'] = planned_and_not_used_dump_area

excavation_analysis_df.loc[excavation_analysis_df.index != 0,'Total Area Excavation Planned and Not Done (in Ha)'] = 0
dump_area_analysis_df.loc[dump_area_analysis_df.index != 0,'Total Area Dump Planned and Inactive (in Ha)'] = 0

excavation_analysis_df.insert(0,'Area Type','Excavation')
dump_area_analysis_df.insert(0,'Area Type','Dump Area')

merged_dump_and_excavation_df = pd.concat([excavation_analysis_df,dump_area_analysis_df],axis=0,ignore_index=True)
merged_dump_and_excavation_df.fillna(0,inplace=True)
os.makedirs(fr'{output_dir}\Excavation_and_Dump_Analysis',exist_ok=True)
merged_dump_and_excavation_df.to_csv(fr'{output_dir}\Excavation_and_Dump_Analysis\Excavation_and_Dump_Analysis.csv',index=False)


check_and_clip_outside_lease()



    



    




    
























    



























