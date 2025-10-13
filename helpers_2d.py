import os
import geopandas as gpd
import pandas as pd






def delete_duplicates(input_path):
    gdf = gpd.read_file(input_path)
    gdf = gdf.drop_duplicates(subset='geometry')
    gdf = gdf.reset_index(drop=True)
    return gdf



def actual_area_gdf_explode_and_add_area_column(gdf):
    gdf = gdf.explode()
    gdf = gdf.reset_index(drop=True)
    gdf['Area_Name'] = 'Area ' + (gdf.index+1).astype(str)
    gdf['Area_Ha'] = gdf.area / 10000
    gdf = gdf[['Area_Name','Area_Ha','geometry']]
    return gdf


def create_gdf_from_each_feature(actual_gdf, planned_gdf):
    separate_gdfs = [actual_gdf.iloc[[i]] for i in range(len(actual_gdf))]
    outside_lease_boundary_gdfs_list = []
    planned_and_done_gdfs_list = []
    unplanned_and_done_gdfs_list = []
    for index,area_gdf in enumerate(separate_gdfs):
        name_for_area = area_gdf['Area_Name']
        outside_lease_boundary_name_gdf = name_for_area + '_outside_lease_boundary_gdf'
        planned_and_done_name_gdf = name_for_area + '_planned_and_done_gdf'
        unplanned_and_done_name_gdf = name_for_area + '_unplanned_and_done_gdf'
        

        outside_lease_boundary_name_gdf = gpd.overlay(area_gdf, mine_boundary_gdf,how='difference')
        outside_lease_boundary_name_gdf = outside_lease_boundary_name_gdf.dissolve(by='Area_Name')
        outside_lease_boundary_name_gdf['Area_Name'] = 'Area ' + str(index +1)
        outside_lease_boundary_name_gdf['Area_Ha'] = (outside_lease_boundary_name_gdf.area.values /10000).round(2)


        outside_lease_boundary_gdfs_list.append(outside_lease_boundary_name_gdf)

        planned_and_done_name_gdf = gpd.overlay(area_gdf,planned_gdf,how='intersection')
        planned_and_done_name_gdf = planned_and_done_name_gdf.dissolve()
        planned_and_done_name_gdf['Area_Name'] = 'Area ' + str(index +1)
        planned_and_done_name_gdf['Area_Ha'] = (planned_and_done_name_gdf.area.values /10000).round(2)


        planned_and_done_gdfs_list.append(planned_and_done_name_gdf)

        unplanned_and_done_name_gdf = gpd.overlay(area_gdf,planned_gdf,how='difference')
        unplanned_and_done_name_gdf = unplanned_and_done_name_gdf.dissolve()
        unplanned_and_done_name_gdf['Area_Name'] = 'Area ' + str(index +1)
        unplanned_and_done_name_gdf['Area_Ha'] = (unplanned_and_done_name_gdf.area/10000).round(2)

        unplanned_and_done_gdfs_list.append(unplanned_and_done_name_gdf)


    return outside_lease_boundary_gdfs_list, planned_and_done_gdfs_list, unplanned_and_done_gdfs_list



#Merging Geodataframes
def merge_geodataframe(list_of_gdf, pit_or_dump, planned_or_not):
    merged_gdf = gpd.GeoDataFrame(pd.concat(list_of_gdf,ignore_index=True))   
    return merged_gdf



#Creating Shapefiles
def create_shapefiles(gdf, shapefile_name, max_no_of_features, pit_or_dump, planned_or_not):
    os.makedirs(fr'{output_dir}\Shapefile_Outputs',exist_ok=True)
    os.makedirs(fr'{output_dir}\GeoJSON_Outputs',exist_ok=True)
    if len(gdf) > 0:
        df = gdf[['Area_Name','Area_Ha']]
        gdf = gdf.explode()
        gdf.reset_index(inplace=True)
        gdf =  gdf[['Area_Name','Area_Ha','geometry']]
        gdf['Area_Ha'] = gdf['Area_Ha'].round(3)
        if pit_or_dump == 'Pit' and planned_or_not == 'Planned':
            gdf['Section_Name'] = 'PEA_' + (gdf.index+1).astype(str)
            print(gdf.index)
        elif pit_or_dump == 'Pit' and planned_or_not == 'Unplanned':
            gdf['Section_Name'] = 'UEA_' + (gdf.index+1).astype(str)
        elif pit_or_dump == 'Pit' and planned_or_not == 'Outside':
            gdf['Section_Name'] = 'OLEA_' + (gdf.index+1).astype(str)
        elif pit_or_dump == 'Dump' and planned_or_not == 'Planned':
            gdf['Section_Name'] = 'PDA_' + (gdf.index+1).astype(str)
        elif pit_or_dump == 'Dump' and planned_or_not == 'Unplanned':
            gdf['Section_Name'] = 'UDA_' + (gdf.index+1).astype(str)
        elif pit_or_dump == 'Dump' and planned_or_not == 'Outside':
            gdf['Section_Name'] = 'OLDA_' + (gdf.index+1).astype(str)
        elif pit_or_dump == 'Dump' and planned_or_not == 'Not_Used':
            pass
        elif pit_or_dump == 'Pit' and planned_or_not == 'Not_Used':
            pass
        else:
            raise ValueError("Wrong Details in Creating GDF")
        gdf.to_file(fr'{output_dir}\Shapefile_Outputs\{shapefile_name}.shp',index=False)
        gdf.to_file(fr'{output_dir}\GeoJSON_Outputs\{shapefile_name}.geojson',index=False)

        return df
        
    elif len(gdf) == 0:
        df = pd.DataFrame()
        df['Area_Name'] = ['Area ' + str(i+1) for i in range(max_no_of_features)]
        df['Area_Ha'] = 0
        return df
    

def correct_area_name(df,area_column_name):
    df = df.rename({'Area_Ha':area_column_name},axis=1)
    return df
    

def merge_dataframes(df_1,df_2,df_3):
    merged_df = pd.merge(pd.merge(df_1,df_2,on='Area_Name',how='outer'),df_3,on='Area_Name',how='outer') 
    return merged_df



#Clipping Outputs If Outside Lease Boundary
def check_and_clip_outside_lease():
    excavation_outside_lease_boundary = fr'{output_dir}\GeoJSON_Outputs\Outside_Lease_Boundary_Excavation.geojson'
    dump_outside_lease_boundary = fr'{output_dir}\GeoJSON_Outputs\Outside_Lease_Boundary_Dump_Area.geojson'

    if os.path.exists(excavation_outside_lease_boundary) == True:
        clipped_unplanned_excavation_gdf = gpd.overlay(unplanned_and_done_excavation_gdf,mine_boundary_gdf,how='intersection')
        clipped_unplanned_excavation_gdf.reset_index(drop=True,inplace=True)
        clipped_unplanned_excavation_gdf['Section_Name'] = 'UEA_' + clipped_unplanned_excavation_gdf.index.astype(str)
        clipped_unplanned_excavation_gdf['Area_Ha'] = clipped_unplanned_excavation_gdf.area/10000
        clipped_unplanned_excavation_gdf = clipped_unplanned_excavation_gdf[['Area_Name','Area_Ha','Section_Name','geometry']]
        os.makedirs(fr'{output_dir}\Clipped_Unplanned_Outputs',exist_ok=True)
        clipped_folder = fr'{output_dir}\Clipped_Unplanned_Outputs'
        clipped_unplanned_dump_area_gdf.to_file(fr'{clipped_folder}\Unplanned_and_Done_Excavation.shp',index=False)

    elif os.path.exists(dump_outside_lease_boundary) == True:
        clipped_unplanned_dump_area_gdf = gpd.overlay(unplanned_and_used_dump_area_gdf, mine_boundary_gdf,how='intersection')
        clipped_unplanned_dump_area_gdf.reset_index(drop=True,inplace=True)
        clipped_unplanned_dump_area_gdf['Section_Name'] = 'UDA_' + clipped_unplanned_dump_area_gdf.index.astype(str)
        clipped_unplanned_dump_area_gdf['Area_Ha'] = clipped_unplanned_dump_area_gdf.area/10000
        clipped_unplanned_dump_area_gdf = clipped_unplanned_dump_area_gdf[['Area_Name','Area_Ha','Section_Name','geometry']]
        os.makedirs(fr'{output_dir}\Clipped_Unplanned_Outputs',exist_ok=True) 
        clipped_folder = fr'{output_dir}\Clipped_Unplanned_Outputs'
        clipped_unplanned_dump_area_gdf.to_file(fr'{clipped_folder}\Unplanned_and_Used_Dump_Area.shp',index=False)