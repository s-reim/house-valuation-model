import pandas as pd
import itertools
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('house_price_data_fyn_distances_EN_with_energylabelencoded_noOHE.csv')

# Select the attributes for similarity comparison
attributes = ['Year of construction', 'Living area', 'Land area', 'No. of toilets and baths',
              'Distance to coast', 'Distance to railroad', 'Distance to motorway',
              'Distance to trafficroads', 'Energylabel encoded']

# Define the weight for X and Y attributes
weight_x = 5000
weight_y = 5000

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[attributes + ['X', 'Y']])

# Compute pairwise distances between houses based on attributes
distances_attributes = pairwise_distances(normalized_data[:, :-2])

# Set the diagonal elements to a large value
np.fill_diagonal(distances_attributes, float('inf'))

# Compute pairwise distances between houses based on X and Y attributes
distances_xy = pairwise_distances(normalized_data[:, -2:])

# Get the indices of the most alike pairs based on attributes and significantly different X and Y values
selected_pairs = []
num_pairs = 10  # Number of pairs to select
for _ in range(num_pairs):
    min_dist = float('inf')
    min_pair = None
    for i, j in itertools.combinations(range(len(data)), 2):
        if (distances_attributes[i, j] < min_dist and
                abs(data.iloc[i]['X'] - data.iloc[j]['X']) >= weight_x and
                abs(data.iloc[i]['Y'] - data.iloc[j]['Y']) >= weight_y):
            min_dist = distances_attributes[i, j]
            min_pair = (i, j)
    if min_pair is not None:
        selected_pairs.append(min_pair)
        distances_attributes[min_pair[0], :] = distances_attributes[:, min_pair[0]] = float('inf')
        distances_attributes[min_pair[1], :] = distances_attributes[:, min_pair[1]] = float('inf')

# Prepare the data for saving
pairs_data = []
for pair in selected_pairs:
    house1 = data.iloc[pair[0]]
    house2 = data.iloc[pair[1]]
    pair_data = house1.append(house2)
    pairs_data.append(pair_data)

# Create a new DataFrame with the selected pairs data
pairs_df = pd.DataFrame(pairs_data)

# Save the selected pairs to a CSV file
pairs_df.to_csv('selected_pairs.csv', index=False)