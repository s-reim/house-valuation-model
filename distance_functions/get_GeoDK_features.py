import geopandas as gpd
import os
from tqdm import tqdm 
import requests

'''
    Calculate distance to various GeoDK features
'''
# Define list of kommunekode for Fyn
fyn_kommunekode = ["0430", "0420", "0440", "0410", "0480", "0450", "0461", "0479"]

# Load shapefile of all municipalities
municipalities = gpd.read_file("./data/dagi_2000m_nohist_l1.kommuneinddeling.shp") # Data pre-downloaded from DAF - can be automized

# Extract Fyn municipalities based on kommunekode
fyn_municipalities = municipalities[municipalities['kommunekod'].isin(fyn_kommunekode)]

# Aggregate based on municipality and create new GeoDataFrame
fyn_agg = fyn_municipalities.dissolve(by='regionskod', aggfunc='sum')

# Save as fyn.shp
fyn_agg.to_file("fyn.shp")

# Read the shapefile using geopandas
shapefile_path = 'fyn.shp' # fyn.shp 
dataframe = gpd.read_file(shapefile_path)

# Get the extent of the shapefile
extent = dataframe.total_bounds

# Print the extent
print("Extent: ", extent) # Use for bbox

data_name = 'Vejmidte' # Insert typename here, e.g. vejmidte, soe, vandløbsmidte, etc...

base_url = "https://api.dataforsyningen.dk/GeoDanmark60_NOHIST_GML3_DAF"
params = {
    "service": "WFS",
    "request": "GetFeature",
    "token": "hidden", # Insert token from dataforsyningen
    "version": "1.1.0",
    "typename": f"{data_name}",
}

bbox = [542686.88, 6089566.93, 617663.54, 6167391.46] # Defined from extent

minx, miny, maxx, maxy = bbox

# Define the size of smaller rectangles
rectangle_size = 5000 

# Calculate the number of rectangles that can fit within the bounding box
num_rectangles = int((maxx - minx) / rectangle_size) * int((maxy - miny) / rectangle_size)

# Iterate over the smaller rectangles
for i in tqdm(range(num_rectangles), total=num_rectangles):
    # Calculate the row and column indices of the current rectangle
    row = i // (int((maxx - minx) / rectangle_size))
    col = i % (int((maxx - minx) / rectangle_size))
    
    # Calculate the coordinates of the smaller rectangle
    rect_minx = minx + col * rectangle_size
    rect_miny = miny + row * rectangle_size
    rect_maxx = rect_minx + rectangle_size
    rect_maxy = rect_miny + rectangle_size
        
    params["bbox"] = f"{rect_minx},{rect_miny},{rect_maxx},{rect_maxy}"
    response = requests.get(base_url, params=params)
    data = response.content
    if data:
        with open('./data/temp/'+data_name+str(i)+'.gml', "ab") as file:
            file.write(data)


merged_gdf = None  # Initialize an empty GeoDataFrame to store the merged data

for i in tqdm(range(num_rectangles), total=num_rectangles):
    file_path = f"'./data/temp/'{data_name}{i}.gml"

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            gdf = gpd.read_file(file_path)
            if merged_gdf is None:
                merged_gdf = gdf
            else:
                merged_gdf = merged_gdf.append(gdf, ignore_index=True)

            # Further processing with the GeoDataFrame
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

shapefile_name = data_name+'.shp'
merged_gdf.to_file(shapefile_name)