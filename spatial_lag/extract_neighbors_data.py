import pandas as pd
import aiohttp
import asyncio
import csv
from tqdm import tqdm

'''
    Step 3: Retrieve data for neighbors
'''

valid_values = ['120', '910', '920', '930', '940', '950', '960']

df_sales = pd.read_csv("salgspriser.csv", sep=';', usecols=['KOMMUNE_NR', 'EJD_NR', 'OVERDRAGELSES_KODE_T', 'KOEBESUM_BELOEB', 'MODTAGELSE_DATO'], low_memory=False)

# Preprocessing step
sales_dict = {}
grouped_sales_data = df_sales.loc[
    (df_sales['OVERDRAGELSES_KODE_T'] == 'Almindelig frit salg') & (df_sales['KOEBESUM_BELOEB'] > 10)
].groupby(['KOMMUNE_NR', 'EJD_NR'])

for group, group_data in grouped_sales_data:
    sales_price = group_data['KOEBESUM_BELOEB'].values[0] if not group_data.empty else None
    sales_date = group_data['MODTAGELSE_DATO'].values[0] if not group_data.empty else None
    sales_dict[group] = {'Salgspris': sales_price, 'Salgsdato': sales_date}

async def get_bbr(lokal_id):
    url = f'https://services.datafordeler.dk/BBR/BBRPublic/1/REST/bygning?username=HSFZLLFRSY&password=Test1234&Jordstykke={lokal_id}&format=json'
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                data = await response.json()
                if data:
                    if any('byg021BygningensAnvendelse' in d and d.get('byg021BygningensAnvendelse') not in valid_values for d in data):
                        return None, None, None
                    else:
                        samlet_bolig_areal = 0
                        for bygning in data:
                            if 'byg039BygningensSamledeBoligAreal' in bygning and 'byg404Koordinat' in bygning:
                                samlet_bolig_areal += bygning.get('byg039BygningensSamledeBoligAreal')
                                koordinat = bygning.get('byg404Koordinat')
                                kommunekode = bygning.get('kommunekode').lstrip('0')
                return samlet_bolig_areal, koordinat, kommunekode
        except:
            return None, None, None

async def get_ejdnr(sfeid):
    url = f'https://services.datafordeler.dk/BBR/BBRPublic/1/rest/ejendomsrelation?username=LLAWNWWDHJ&password=Rest01Test-&SamletFastEjendom={sfeid}&format=json'
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                data = await response.json()
                if data:
                    ejendomsnummer = data[0].get("ejendomsnummer")
                return ejendomsnummer
        except:
            return None

async def process_building(item):
    lokal_id = item['LokalID']
    SFEid = item['SFEid']

    samlet_bolig_areal, koordinat, kommunekode = await get_bbr(lokal_id)
    if koordinat:
        try:
            x_y_str = koordinat.replace('POINT(', '').replace(')', '')  # Remove the "POINT(" and ")" parts
            x_y = x_y_str.split()  # Split the remaining string by whitespace
            X_coord = float(x_y[0])
            Y_coord = float(x_y[1])
        except:
            X_coord = None
            Y_coord = None
    else:
        X_coord = None
        Y_coord = None

    ejendomsnummer = await get_ejdnr(SFEid)

    sales_info = sales_dict.get((kommunekode, ejendomsnummer))
    if sales_info is not None:
        salgspris = sales_info['Salgspris']
        salgsdato = sales_info['Salgsdato']
    else:
        salgspris, salgsdato = None, None

    row = {
        "Jordstykke_Id": lokal_id,
        "X": X_coord,
        "Y": Y_coord,
        "Boligareal": samlet_bolig_areal,
        "Salgspris": salgspris,
        "Salgsdato": salgsdato,
    }
    return row


async def main(chunk):
    rows = []
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(process_building(item)) for _, item in chunk.iterrows()]
    with tqdm(total=len(tasks)) as pbar:
        # Iterate over completed tasks
        for completed_task in asyncio.as_completed(tasks):
            row = await completed_task
            rows.append(row)

            # Update the progress bar
            pbar.update(1)
    return rows

chunk_size = 1000
rows = []
# Read the CSV file in chunks/batches
for chunk in pd.read_csv('house_price_data_IDs.csv', chunksize=chunk_size):
    chunk_rows = asyncio.run(main(chunk))
    rows.extend(chunk_rows)

valid_rows = [row for row in rows if row is not None]
print(len(valid_rows))

if len(valid_rows) > 0:
    with open(f"house_price_data_neighbors.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=valid_rows[0].keys())
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)