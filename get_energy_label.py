import requests
import pandas as pd
from pyproj import Proj, transform
import aiohttp
import asyncio
from tqdm import tqdm
import asyncio
import aiohttp
import pandas as pd

# Read the CSV file
df = pd.read_csv('house_price_data_fyn_2.csv')

# Define the coordinate systems
utm32n = Proj(init='epsg:25832')  # UTM 32N
wgs84 = Proj(init='epsg:4326')    # WGS84
url = 'https://emoweb.dk/EMOData/EMOData.svc/GetEnergyLabelFromCoordinatesWithSearchRadius'
headers = {
    'Accept': 'application/json',
    'Authorization': 'hidden'
}

# Convert UTM coordinates to WGS84
lon, lat = transform(utm32n, wgs84, df['X'].values, df['Y'].values)

df['LON'] = lon
df['LAT'] = lat

BATCH_SIZE = 1000

async def fetch_energy_label(session, url, headers, params):
    async with session.get(url, headers=headers, params=params) as response:
        data = await response.json()
        energy_label = data.get('SearchResults')
        if energy_label:
            for x in energy_label:
                if x.get('LabelStatus') == "VALID":
                    label = x.get('EnergyLabelClassification')
                    return label
        else:
            return 'None'

async def process_batch(session, batch):
    tasks = []
    for _, row in batch.iterrows():
        X = row['LAT']
        Y = row['LON']
        url = 'https://emoweb.dk/EMOData/EMOData.svc/GetEnergyLabelFromCoordinatesWithSearchRadius'
        headers = {
            'Accept': 'application/json',
            'Authorization': 'hidden'
        }
        params = {
            'coordinateX': X,
            'coordinateY': Y,
            'pagesize': 100,
            'pageNumber': 1,
            'searchRadius': 0.00018
        }
        task = asyncio.ensure_future(fetch_energy_label(session, url, headers, params))
        tasks.append(task)
    energy_labels = await asyncio.gather(*tasks)
    return energy_labels

async def process_dataframe(df):
        num_rows = len(df)
        num_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(num_batches), desc="Processing batches"):            
            async with aiohttp.ClientSession() as session:
                start = i * BATCH_SIZE
                end = min((i + 1) * BATCH_SIZE, num_rows)
                batch = df.iloc[start:end]

                energy_labels = await process_batch(session, batch)
                df.loc[start:end-1, 'Energylabel'] = energy_labels

# Assuming your DataFrame is named 'df'
loop = asyncio.get_event_loop()
loop.run_until_complete(process_dataframe(df))

df = df.drop(['LON', 'LAT'], axis=1)

# Fill missing values with 'None'
df['Energylabel'].fillna('None', inplace=True)

# Define mapping dictionary
label_mapping = {
    'None': 0,
    'G': 1,
    'F': 2,
    'E': 3,
    'D': 4,
    'C': 5,
    'B': 6,
    'A2010': 7,
    'A2015': 8,
    'A2020': 9
}

# Apply mapping and create new column
df['Energylabel encoded'] = df['Energylabel'].map(label_mapping)

# Export the updated dataframe to a new CSV file
df.to_csv('house_price_data_final.csv', index=False)