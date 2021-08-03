# %%
from pathlib import Path
import re
import os
import pandas as pd
# %%
df_path = Path("./Resources/10K_10Q_List.csv")
df = pd.read_csv(df_path)
df.head()
# %%
for filing in os.listdir("./Resources/Filings"):
    if filing.endswith(".txt"):
        filing_path = Path(os.path.join("./Resources/Filings",filing))
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
        output_path = Path('./Resources/Cleaned Filings/' + filing)
        file = open(output_path, 'w+')
        file.write(result)
        file.close()
        os.remove(filing_path)
# %%
