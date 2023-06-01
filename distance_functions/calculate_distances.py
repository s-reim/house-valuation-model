import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points

data = pd.read_csv('fyn_house_price_data.csv')

# Load the coastline shapefile as a GeoDataFrame
railroad = gpd.read_file("./geodk_data/Jernbane.shp")
Vandloebsmidte = gpd.read_file("./geodk_data/Vandloebsmidte.shp")
windturbines = gpd.read_file("./geodk_data/Vindmoller.shp")
forest = gpd.read_file("./geodk_data/Skov.shp")
lake = gpd.read_file("./geodk_data/soe.shp")
roads = gpd.read_file("./geodk_data/Vejmidte.shp")

large_roads = roads[roads['vejkategor'].isin(['Hovedrute'])]
traffic_roads = roads[roads['vejkategor'].isin(['Gennemfartsrute', 'Fordelingsrute'])]

geometry = gpd.points_from_xy(data['X'], data['Y'])
gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=lake.crs)

# Define a function to calculate the distance from a point to the nearest point on the coast
def distance_to_railroad(point):
    nearest_edge_point = railroad.geometry.boundary.distance(point).idxmin()
    nearest_edge_point = railroad.geometry.boundary[nearest_edge_point]
    return point.distance(nearest_edge_point)

def distance_to_stream(point):
    nearest_edge_point = Vandloebsmidte.geometry.boundary.distance(point).idxmin()
    nearest_edge_point = Vandloebsmidte.geometry.boundary[nearest_edge_point]
    return point.distance(nearest_edge_point)

# Define a function to calculate the distance from a point to the nearest wind turbine
def distance_to_windturbines(point):
    nearest_turbine = windturbines.geometry.unary_union.nearest_point(point)
    return point.distance(nearest_turbine)

# Define a function to calculate the distance from a point to the nearest forest
def distance_to_forest(point):
    nearest_forest = forest.geometry.unary_union.nearest_point(point)
    return point.distance(nearest_forest)

# Define a function to calculate the distance from a point to the nearest lake
def distance_to_lake(point):
    nearest_lake = lake.geometry.unary_union.nearest_point(point)
    return point.distance(nearest_lake)

# Apply the function to each point in the GeoDataFrame to calculate the distance to feature
gdf['distance_to_railroad'] = gdf.geometry.apply(distance_to_railroad)
gdf['distance_to_stream'] = gdf.geometry.apply(distance_to_stream)
gdf['distance_to_windturbines'] = gdf.geometry.apply(distance_to_windturbines)
gdf['distance_to_forest'] = gdf.geometry.apply(distance_to_forest)
gdf['distance_to_lake'] = gdf.geometry.apply(distance_to_lake)
gdf['distance_to_motorway'] = gdf.geometry.apply(lambda x: x.distance(large_roads.unary_union))
gdf['distance_to_trafficroads'] = gdf.geometry.apply(lambda x: x.distance(traffic_roads.unary_union))
# Add more here...

gdf_without_geometry = gdf.drop('geometry', axis=1)
gdf_without_geometry.to_csv('house_price_data_fyn_distances.csv', index=False)
