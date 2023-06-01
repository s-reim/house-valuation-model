import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm

'''
    The following script can be seen as step 2 followed by "extract_neighbors_data.py". This script is a pre-processing step, in which
    SFEid and LokalId for neighbors is retrieved. The script is a smart way of finding neighbors by searching within a bbox from each house.
    In this way we don't have to retrieve ALL houses first and then find the nearest to each.
'''

async def get_ids(north, east, south, west):
    url = f"https://services.datafordeler.dk/MATRIKEL/MatrikelGaeldendeOgForeloebigWFS/1.0.0/WFS?username=QYZVHCVJUC&password=Prod01_Secure&version=1.1.0&service=wfs&request=GetFeature&typename=Jordstykke_Gaeldende&srsname=25832&bbox={west},{north},{east},{south}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "xml")
                features = soup.find_all("mat:Jordstykke_Gaeldende")
                lokal_ids = []
                samletfastejendomids = []
                for feature in features:
                    lokal_ids.append(feature.find("mat:id.lokalId").text)
                    samletfastejendomids.append(feature.find("mat:samletFastEjendomLokalId").text)
                return lokal_ids, samletfastejendomids
        except:
            return None, None

async def process_building(item):
    north = item['North']
    south = item['South']
    east = item['East']
    west = item['West']

    lokal_ids, samletfastejendomsids = await get_ids(north, east, south, west)
    return lokal_ids, samletfastejendomsids


async def main(chunk):
    id_list = []

    async def process_and_append(building):
        lokal_ids, samletfastejendomsids = await process_building(building)
        if lokal_ids is not None and samletfastejendomsids is not None:
            pairs = list(zip(lokal_ids, samletfastejendomsids))
            id_list.extend(pairs)

    total_buildings = len(chunk)
    with tqdm(total=total_buildings) as pbar:
        tasks = []
        for _, building in chunk.iterrows():
            task = asyncio.create_task(process_and_append(building))
            task.add_done_callback(lambda _: pbar.update(1))
            tasks.append(task)
        await asyncio.gather(*tasks)

    return id_list

chunk_size = 1000
rows = []
# Read the CSV file in chunks/batches
for chunk in pd.read_csv('fyn_house_price_data.csv', chunksize=chunk_size): # fyn_house_price_data is a concatenation of all house_price_data_{municipal_code} - I used glob for this
    # Add the directions to the chunk dataframe
    chunk['North'] = chunk['Y'] + 200
    chunk['South'] = chunk['Y'] - 200
    chunk['East'] = chunk['X'] + 200
    chunk['West'] = chunk['X'] - 200
    
    chunk_rows = asyncio.run(main(chunk))
    rows.extend(chunk_rows)


valid_rows = [row for row in rows if all(value is not None for value in row)]
print(len(valid_rows))

valid_rows1 = list(set(valid_rows)) # Remove duplicates
print(len(valid_rows1))

field_names = ['LokalID', 'SFEid']

if len(valid_rows1) > 0:
    with open("house_price_data_IDs.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(field_names)
        writer.writerows(valid_rows)