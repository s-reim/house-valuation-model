import pandas as pd

'''
    The following script performs price projection using data from Danmarks Statistik. Input/output can be changed between neighbors data and primary data
'''
# Step 1: Remove rows with NaN values
data = pd.read_csv('house_price_data_neighbors.csv')
data = data.dropna()

# Step 2: Extract the year from Salgsdato and fetch the average index from the provided website
data['SalgsYear'] = data['Salgsdato'].str.extract('(\d{4})')
data = data[data['SalgsYear'] != '2023']

price_index = {
    '1992': 34.55,
    '1993': 34.17,
    '1994': 38.11,
    '1995': 40.9,
    '1996': 47.325,
    '1997': 51.075,
    '1998': 58.95,
    '1999': 61.375,
    '2000': 63.925,
    '2001': 65.325,
    '2002': 65.45,
    '2003': 66.225,
    '2004': 70.525,
    '2005': 81.325,
    '2006': 99,
    '2007': 110.4,
    '2008': 111.1,
    '2009': 101.25,
    '2010': 100.5,
    '2011': 96.4,
    '2012': 91.625,
    '2013': 92.35,
    '2014': 94.725,
    '2015': 98.625,
    '2016': 100.9,
    '2017': 104.175,
    '2018': 109.85,
    '2019': 113.875,
    '2020': 118.15,
    '2021': 129.975,
    '2022': 128.45
}

data['Salgspris_Adjusted'] = data.apply(lambda row: row['Salgspris'] * (price_index['2022'] / price_index[row['SalgsYear']]), axis=1)

data.to_csv('neighbors_price_adjusted.csv', index=False)