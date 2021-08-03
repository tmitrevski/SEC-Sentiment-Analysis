# %%
import pandas as pd
import re
import os
from pathlib import Path
import requests
import time

# %%
file_path = Path("Resources/10K_10Q_List.csv")
df = pd.read_csv(file_path)
df.head()
# %%
headers = {'user-agent': 'project_10k/0.0.1'}
for index,filing in enumerate(df.Filename[242910:]):
    requestURL = filing
    filing_path = Path('./Resources/Filings/' + str(index+242910) + '.txt')
    response_data = requests.get(requestURL, headers = headers)
    file = open(filing_path, 'w+')
    file.write(response_data.text)
    file.close()
    file = open(filing_path)
    line = file.read().replace("\n", " ")
    file.close()
    result = re.search('<TYPE>10-K(.*?)</TEXT>', line)
    if result == None:
        result = re.search('<TYPE>10-Q(.*?)</TEXT>', line)
    result = re.sub("((?i)<TYPE>).*?(?=<)", "", result.group(1))
    result = re.sub("((?i)<SEQUENCE>).*?(?=<)", "", result)
    result = re.sub("((?i)<FILENAME>).*?(?=<)", "", result)
    result = re.sub("((?i)<DESCRIPTION>).*?(?=<)", "", result)
    result = re.sub("(?s)(?i)<head>.*?</head>", "", result)
    result = re.sub("(?s)(?i)<(table).*?(</table>)", "", result)
    result = re.sub("(?s)(?i)(?m)> +Item|>Item|^Item", ">°Item", result) 
    result= re.sub("(?s)<.*?>", " ", result)
    result = re.sub("&(.{2,6});", " ", result)
    result = re.sub("(?s) +", " ", result)
    result = re.sub("^.*?UNITED STATES SECURITIES AND EXCHANGE COMMISSION", "UNITED STATES SECURITIES AND EXCHANGE COMMISSION", result)
    output_path = Path('./Resources/Cleaned Filings/' + str(index+242910) + '.txt')
    file = open(output_path, 'w+')
    file.write(result)
    file.close()
    os.remove(filing_path)
# %%
