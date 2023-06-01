import asyncio
import aiohttp
import json
import pandas as pd
import csv
from tqdm import tqdm

''' DATA COLLECTION SCRIPT
    The following is a script compiled during a semester project in fall 2023 at Aalborg University (AAU)
    The script retrieves various property data for every single-family house on Fyn sold in the period 2016-2022. 
    Some data cleaning happens during the script, such as sorting out co-valuation properties and single-family houses 
    located on agricultural or commercial properties.  
    asyncio has been utilized to speed up the calls to the different servers. Total run time: approx. 30 minutes.
'''

# Define list of BBR building usage codes which are linked to single-family house properties, such that it can be checked whether
# the single-family houses lay upon a plot of land where buildings not in line with the list is present. 
valid_values = ['120', '910', '920', '930', '940', '950', '960'] 

# The following code defines two dictionaries containing respectively SVUR valuations and sales
# In this way we can get the official valuation and sales price of a property by looking it up using municipal and property code.
df_sales = pd.read_csv("salgspriser.csv", sep=';', usecols=['KOMMUNE_NR', 'EJD_NR', 'OVERDRAGELSES_KODE_T', 'KOEBESUM_BELOEB', 'MODTAGELSE_DATO'], low_memory=False)
df_vur = pd.read_csv("vurderinger.csv", sep=';', usecols=['KOMMUNE_NR', 'EJD_NR', 'VUR_AAR', 'EJD_VAERDI'], low_memory=False)
df_vur = df_vur.sort_values(by='VUR_AAR', ascending=False) # Sort by year, newest first
df_vur.drop_duplicates(subset=['KOMMUNE_NR', 'EJD_NR'], keep='first', inplace=True) # Only keep newest valuation for each property
vurdering_dict = {(k[0], k[1]): v for k, v in zip(df_vur[['KOMMUNE_NR', 'EJD_NR']].values, df_vur['EJD_VAERDI'].values)} 

sales_dict = {}
sales_data = df_sales.loc[(df_sales['OVERDRAGELSES_KODE_T'] == 'Almindelig frit salg') & (df_sales['KOEBESUM_BELOEB'] > 10)] # Filter the sales, such that only "Ordinary free sale" above 10 DKK is used

for row in sales_data.itertuples(index=False):
    kommune_nr = row.KOMMUNE_NR
    ejd_nr = row.EJD_NR
    salgspris = row.KOEBESUM_BELOEB
    modtagelse_dato = row.MODTAGELSE_DATO # We save the sales date to perform price projection later
    if (kommune_nr, ejd_nr) not in sales_dict:
        sales_dict[(kommune_nr, ejd_nr)] = {'salgspris': salgspris, 'modtagelse_dato': modtagelse_dato}
    else:
        # If there are multiple sales of the same property, use the latest one
        if modtagelse_dato > sales_dict[(kommune_nr, ejd_nr)]['modtagelse_dato']:
            sales_dict[(kommune_nr, ejd_nr)]['salgspris'] = salgspris
            sales_dict[(kommune_nr, ejd_nr)]['modtagelse_dato'] = modtagelse_dato

# The retrieval of data follows. First, a list of municipal codes to retrieve data for:
komk_values = ["0430", "0420", "0440", "0410", "0480", "0450", "0461", "0479"]
for komk in komk_values:
    # Sales price projection DEPRECATED - SEE price_adjuster.py
    async def adjust_price(sales_dict, kommune_nr, ejd_nr):
        sale_info = sales_dict.get((str(kommune_nr), float(ejd_nr)))
        try:
            modtagelse_dato = int(sale_info['modtagelse_dato'].split()[0])
            if modtagelse_dato > 2015 and modtagelse_dato < 2023:
                salgspris = sale_info['salgspris'] * (1 + 0.0237) ** (2022 - modtagelse_dato) # SIMPLE COMPOUND FORMULA - THIS IS DEPRECATED. Perform price projection using "price_adjuster.py"
            else:
                return None, None
            return modtagelse_dato, salgspris
        except:
            return None, None

    # Plot of land area retrieved through MAT - with agricultural filtering
    async def get_grundareal(jordstykke):
        grundarealurl = f"https://services.datafordeler.dk/Matrikel/Matrikel/1/REST/SamletFastEjendom?username=hidden&password=hidden&JordstykkeId={jordstykke}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(grundarealurl) as response:
                    grundareal_data = await response.json()
                    if grundareal_data:
                        try:
                            jordstykkeresponse1 = grundareal_data.get("features")[0].get("properties")
                            if jordstykkeresponse1.get("landbrugsnotering") != None: # If the plot is defined as agricultural then remove from data
                                return None
                            jordstykkeresponse = jordstykkeresponse1.get("jordstykke")[0].get("properties")
                            grundareal = int(jordstykkeresponse.get("registreretAreal")) - int(jordstykkeresponse.get("vejareal"))
                            return grundareal
                        except:
                            return None
            except:
                return None

    # Function for retrieving number of toilets and baths from BBR unit
    async def get_toilet_bad(bygningsid):
        toileturl = f'https://services.datafordeler.dk/BBR/BBRPublic/1/REST/enhed?username=hidden&password=hidden&Bygning={bygningsid}&status=6'
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(toileturl) as response:
                    toilet_data = await response.json()
                    if toilet_data:
                        try:
                            antaltoilet = toilet_data[0]['enh065AntalVandskylledeToiletter']
                            antalbad = toilet_data[0]['enh066AntalBadeværelser']
                            toilet_bad = int(antaltoilet)+int(antalbad) # It may be more suitable to return antaltoilet and antalbad
                            return toilet_bad
                        except:
                            return None
                    else:
                        return None
            except:
                return None

    # Function for retrieving BFE number to use for identification of property
    async def get_bfenummer(husnummer, jordstykke):
        bfenummer_url = f"https://services.datafordeler.dk/BBR/BBRPublic/1/REST/grund?username=hidden&password=hidden&Husnummer={husnummer}&format=json"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(bfenummer_url) as response:
                    bfenummer_data = await response.json()
                    if bfenummer_data:
                        for element in bfenummer_data:
                            if jordstykke in element.get("jordstykkeList"):
                                bfenummer = element.get("bestemtFastEjendom").get("bfeNummer")
                                return bfenummer
            except:
                pass

    # Function for retrieving propertynumber from BBR using BFE number
    async def get_ejendomsnummer(bfenummer):
        ejendomsnummer_url = f'https://services.datafordeler.dk/BBR/BBRPublic/1/rest/ejendomsrelation?username=hidden&password=hidden&BFENummer={bfenummer}&format=json'
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(ejendomsnummer_url) as response:
                    ejendomsnummer_data = await response.json()
                    if ejendomsnummer_data:
                        ejendomsnummer = ejendomsnummer_data[0].get("ejendomsnummer")
                        return ejendomsnummer
            except:
                pass
    
    # Filter function to ensure the property at hand is not located on a plot of land containing buildings which are not associated with single-family properties
    async def validate_building(jordstykke_id):
        validation_url = f'https://services.datafordeler.dk/BBR/BBRPublic/1/REST/bygning?username=hidden&password=hidden&Jordstykke={jordstykke_id}'
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(validation_url) as response:
                    validate_data = await response.json()
                    if validate_data:
                        if any('byg021BygningensAnvendelse' in d and d['byg021BygningensAnvendelse'] not in valid_values for d in validate_data): # Valid values is defined at line 18
                            return True
                        else:
                            return False
                    else:
                        True
            except:
                return True
    
    # Filter function to ensure property at hand is not co-valued.
    async def get_samvurderet(bfenummer):
        validation_url = f'https://services.datafordeler.dk/Ejendomsvurdering/Ejendomsvurdering/1/rest/HentEjendomsvurderingerForBFE?username=hidden&password=hidden&BFENummer={bfenummer}&format=json'
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(validation_url) as response:
                    validate_data = await response.json()
                    if validate_data:
                        most_recent_dict = max(validate_data, key=lambda x: x['år'])
                        if len(most_recent_dict.get('BFEnummerList')) > 1:
                            return True
                        else:
                            return False
                    else:
                        return False
            except:
                return True

    # Primary function to retrieve property specific data
    async def process_json(item):
        jordstykke = item.get('jordstykke')
        validation = await validate_building(jordstykke) # Co-valuation validation
        if validation == True:
            return
        kommunekode = item.get("kommunekode")
        boligareal = item.get("byg039BygningensSamledeBoligAreal")
        boligalder = item.get("byg026Opførelsesår")
        ombygningsalder = item.get("byg027OmTilbygningsår")
        ydervægsmateriale = item.get("byg032YdervæggensMateriale")
        tagtype = item.get("byg033Tagdækningsmateriale")
        varmekilde = item.get("byg056Varmeinstallation")
        supplerende_varmekilde = item.get("byg058SupplerendeVarme")
        husnummer = item.get('husnummer')
        bygningsid = item.get('id_lokalId')
        koordinat = item.get("byg404Koordinat")
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

        task1 = asyncio.create_task(get_grundareal(jordstykke))
        task3 = asyncio.create_task(get_toilet_bad(bygningsid))
        results = await asyncio.gather(task1, task3)

        grundareal = results[0]
        if not grundareal:
            return
        
        bfenummer = await get_bfenummer(husnummer, jordstykke)

        ejendomsnummer = None
        if bfenummer:
            validate_samvurderet = await get_samvurderet(bfenummer)
            if validate_samvurderet == True:
                return
            ejendomsnummer = await get_ejendomsnummer(bfenummer)
        else:
            return
        if not ejendomsnummer:
            return

        toilet_bad = results[1]

        new_kommune = kommunekode.lstrip('0') # SVUR does not use leading zeros and is therefore removed
        vurdering = vurdering_dict.get((int(new_kommune), int(ejendomsnummer)))
        salgsdato, salgspris = await adjust_price(sales_dict, new_kommune, ejendomsnummer)

        row = {
            "X": X_coord,
            "Y": Y_coord,
            "Boligalder": boligalder,
            "Boligareal": boligareal,
            "Ombygningsalder": ombygningsalder,
            "Grundareal": grundareal,
            "Vurdering": vurdering,
            "Salgspris": salgspris,          
            "Ydervægsmateriale": ydervægsmateriale,
            "Tagtype": tagtype,
            "Varmekilde": varmekilde,
            "Supplerende_varmekilde": supplerende_varmekilde,
            "Toilet_bad": toilet_bad,
            "Jordstykke_id": jordstykke,
            "BFEnummer": bfenummer,
            "Kommunekode": kommunekode,
            "Ejendomsnummer": ejendomsnummer,
            "Salgsdato": salgsdato,
        }
        return row

    # Function to retrieve ALL single-family buildings for the municipal
    async def get_data(komk):
        url = "https://services.datafordeler.dk/BBR/BBRPublic/1/REST/bygning"
        params = {
            "username": "hidden",
            "password": "hidden",
            "Kommunekode": f"{komk}",
            "format": "json",
            "pagesize": 9999999, # Unfortunately we have to retrieve ALL buildings and then sort.. BBR has been notified of this "flaw".
            "status": 6, # Only constructed and not demolished houses is retrieved
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
        print(f'Total number of buildings: {len(data)}') 
        filtered_data = [element for element in data if "byg021BygningensAnvendelse" in element and element["byg021BygningensAnvendelse"] == "120" and "jordstykke" in element and "id_lokalId" in element and "husnummer" in element]
        return filtered_data

    # Function setting up the asynchronous environment
    async def main(filtered_data):
        rows = []
        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(process_json(item)) for item in filtered_data]
        for completed_task in asyncio.as_completed(tasks):
                row = await completed_task
                rows.append(row)
        return rows
    
    # Get all single-family houses for the municipality
    filtered_data = asyncio.run(get_data(komk))
    print(f'Number of buildings after basic filtering: {len(filtered_data)}')

    # To minimize time-outs we run the asynchronous calls in batches of 1000 buildings - approx 6000 asynchronous calls per batch
    chunk_size = 1000
    rows = []
    for i in tqdm(range(0, len(filtered_data), chunk_size)):
        chunk = filtered_data[i:i+chunk_size]
        chunk_rows = asyncio.run(main(chunk))
        rows.extend(chunk_rows)

    valid_rows = [row for row in rows if row is not None] # I'm 99% sure this is redundant now, but for safety we introduced it - removes rows which are only null values
    print(len(valid_rows)) # Just to see amount of buildings retrieved per municapal

    if len(valid_rows) > 0:
        with open(f"house_price_data_{komk}.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=valid_rows[0].keys())
            writer.writeheader()
            for row in valid_rows:
                writer.writerow(row)