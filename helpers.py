import geopandas as gpd
from shapely.geometry import LineString, Point
import os
import rasterio
import shapely
from shapely.ops import split
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import Vec2




##### interpolate the line 
def interpolate_shapefile(input_path: str, output_path: str, interval_cm: float = 1.0):
    """
    Interpolates a shapefile's LineString geometries into points spaced by given interval in centimeters,
    and exports the result as a new shapefile.

    Parameters:
    - input_path (str): Path to the input shapefile.
    - output_path (str): Path where the interpolated shapefile will be saved.
    - interval_cm (float): Distance between points in centimeters (default = 1.0).
    """

    # Convert cm to meters
    interval = interval_cm / 100.0
    print()

    # Load the shapefile
    gdf = gpd.read_file(input_path)

    # Reproject to UTM if in geographic coordinates (degrees)
    if gdf.crs is None or gdf.crs.is_geographic:
        print("Input is in geographic CRS. Reprojecting to UTM Zone 43N (EPSG:32643)...")
        gdf = gdf.to_crs(epsg=32643)  # Change this EPSG code to your region if needed

    interpolated_points = []

    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            length = geom.length
            num_points = int(length // interval) + 1
            distances = [i * interval for i in range(num_points + 1)]
            points = [geom.interpolate(d) for d in distances]
            interpolated_points.extend(points)
        elif isinstance(geom, Point):
            interpolated_points.append(geom)
        else:
            print(f"Skipping unsupported geometry: {type(geom)}")

    # Create GeoDataFrame of points
    interpolated_gdf = gpd.GeoDataFrame(geometry=interpolated_points, crs=gdf.crs)

    # Export to shapefile
    interpolated_gdf.to_file(output_path)

    return interpolated_gdf





def clip_lines_by_boundary(line_shapefile: str, boundary_shapefile: str, output_shapefile: str):
    """
    Clips a line shapefile using a polygon boundary shapefile based on custom rules:
    1. If both ends intersect the polygon, clip and keep only the part inside.
    2. If one end intersects and the other is inside, clip from one side and retain the inside segment.
    3. If both ends are inside (no intersection), keep line as-is.
    """

    # Load shapefiles
    lines_gdf = gpd.read_file(line_shapefile)
    boundary_gdf = gpd.read_file(boundary_shapefile)

    # Reproject to common CRS if needed
    if lines_gdf.crs != boundary_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(lines_gdf.crs)

    boundary_union = boundary_gdf.unary_union

    result_geometries = []
    result_attributes = []

    for idx, row in lines_gdf.iterrows():
        line = row.geometry

        start_pt = Point(line.coords[0])
        end_pt = Point(line.coords[-1])

        start_inside = boundary_union.contains(start_pt)
        end_inside = boundary_union.contains(end_pt)
        line_inside = boundary_union.contains(line)
        line_intersects = line.crosses(boundary_union) or line.touches(boundary_union)

        if line_inside:
            # Entire line is inside polygon (Case 3)
            result_geometries.append(line)
            result_attributes.append(row)

        elif line_intersects:
            # Line crosses or touches polygon boundary
            if not start_inside and not end_inside:
                # Case 1: Line enters and exits polygon — keep only inside segment
                clipped = line.intersection(boundary_union)
                if isinstance(clipped, (LineString, shapely.MultiLineString)):
                    result_geometries.append(clipped)
                    result_attributes.append(row)
            else:
                # Case 2: One end inside, one outside — keep partial segment
                split_parts = split(line, boundary_union.boundary)
                for part in split_parts.geoms:
                    if boundary_union.contains(Point(part.coords[0])) or boundary_union.contains(Point(part.coords[-1])):
                        result_geometries.append(part)
                        result_attributes.append(row)

    # Create output GeoDataFrame
    clipped_gdf = gpd.GeoDataFrame(result_attributes, geometry=result_geometries, crs=lines_gdf.crs)

    # Save to file
    clipped_gdf.to_file(output_shapefile)

    return clipped_gdf


### fetch the elevation on the line 
def elevation_at_point(dtm_src, x, y):
    """Query elevation value from DTM raster at (x, y) location."""
    q_ele = list(dtm_src.sample([(x, y)]))[0][0]
    return q_ele

def get_elevation_points_from_lines(dtm_path: str, clipped_line_shapefile: str, interval: float = 0.01):
    """
    For each point interpolated along clipped lines, fetch elevation from DTM.
    
    Parameters:
    - dtm_path (str): Path to the DTM raster (GeoTIFF).
    - clipped_line_shapefile (str): Path to the clipped line shapefile.
    - interval (float): Spacing between interpolated points (in meters). Default is 0.01 (1 cm).
    
    Returns:
    - List of (x, y, z) tuples.
    """
    # Load line geometries
    lines_gdf = gpd.read_file(clipped_line_shapefile)
    print(lines_gdf)

    # Open DTM raster
    with rasterio.open(dtm_path) as dtm_src:
        # Reproject lines to match DTM CRS if needed
        if lines_gdf.crs != dtm_src.crs:
            print("Reprojecting line geometries to match DTM CRS...")
            lines_gdf = lines_gdf.to_crs(dtm_src.crs)

        points_with_elevation = []

        # Loop through each LineString
        for geom in lines_gdf.geometry:
            if isinstance(geom, LineString):
                length = geom.length
                num_points = int(length // interval) + 1
                distances = [i * interval for i in range(num_points + 1)]
                print(distances)

                for d in distances:
                    pt = geom.interpolate(d)
                    x, y = pt.x, pt.y
                    z = elevation_at_point(dtm_src, x, y)
                    print(z)
                    points_with_elevation.append((x, y, z))
        print(points_with_elevation)
        return points_with_elevation
