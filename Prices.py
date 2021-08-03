# %%
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import requests
import time
from datetime import datetime, timedelta
import yfinance as yf
import json
# %%
file_path = Path("Resources/10K_10Q_List_With_Tickers.csv")
df = pd.read_csv(file_path)
df.head()
# %%
headers = {'user-agent': 'project_10k/0.0.1'}
ticker_url = 'https://www.sec.gov/files/company_tickers_exchange.json'
response_data = requests.get(ticker_url).json()

# %%
print(type(response_data))
# %%
print(response_data['data'][0][3])
# %%
df["Ticker"] = ""
df["Exchange"] = ""
df.head()
# %%
df["Company Name"][0]
# %%
for i in response_data['data']:
    df.loc[df['CIK'] == i[0], ['Ticker','Exchange']] = [i[2], i[3]]
df.head()
# %%
output_path = Path('./Resources/10K_10Q_List_With_Tickers.csv')
df.to_csv(output_path)
# %%
df['Date Filed']= pd.to_datetime(df['Date Filed'])
# %%
df['Ticker'].replace('', np.nan, inplace=True)
df['Exchange'].replace('', np.nan, inplace=True)
df.dropna(inplace=True)
df.head()
# %%
yf_data, missing = [], []
pull_data = False
#interim_output_path = Path('./Resources/yf_data.csv')
for i, (symbol, dates) in enumerate(df.groupby('Ticker')['Date Filed'], 1):
    
    ticker = yf.Ticker(symbol)
    for filing, date in dates.to_dict().items():
        if symbol == 'VCRA' and filing == 46043:
            pull_data = True
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Pull Data Started at ", current_time)
            continue
        if pull_data == True:
            start = date - timedelta(days=93)
            end = date + timedelta(days=31)
            try:
                returned_df = ticker.history(start=start, end=end)
            except json.decoder.JSONDecodeError:
                returned_df = pd.DataFrame()
            if returned_df.empty:
                missing.append(symbol)
            else:
                yf_data.append(returned_df.assign(ticker=symbol, filing=filing))
                #yf_data_interim = pd.concat(yf_data).rename(columns=str.lower)
                #yf_data_interim.to_csv(interim_output_path)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Data Successfully Pulled at ", current_time)
# %%
yf_data = []
# %%
interim_output_path = Path('./Resources/yf_data.csv')
# %%
yf_data[-3]
# %%
yf_data_interim = pd.concat(yf_data).rename(columns=str.lower)
#%%
yf_data_interim2.to_csv(interim_output_path)
# %%
yf_df = pd.read_csv(interim_output_path, index_col='Date', parse_dates=True, infer_datetime_format=True)
#%%
yf_data_interim2
# %%
yf_data_interim2 = pd.concat([yf_df,yf_data_interim]).rename(columns=str.lower)
# %%
