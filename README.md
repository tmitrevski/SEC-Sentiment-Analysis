# SEC-Sentiment-Analysis
This project looks at the contents of 10-K and 10-Q statements to determine if the sentiment contained in the text will predict a shift in the prices during the week after the statement is released.
·  	Sections focused on are 1, 1a, 7 and 7a.
·  	Documents are retrieved from the EDGAR SEC site and include all 10-Ks and 10-Qs released since 2012 for NYSE, NASDAQ and OTC stocks.
·  	Combinations of words are looked at using an N-GRAM model (2 to 4).
·  	The Yahoo Finance API is used to retrieve stock tickers 90 days prior and 30 days after a filing in order to see if the filing influenced the stock prices.
·  	This project then implements a deep RNN in order to predict prices and evaluates the Mean Square Error to check if the price prediction is accurate.


Resources folder can be found at XXXX and all filed need to be unzipped and added to a google drive for use.
Model is contained in the SECNLP.ipynb file and must be run using a GPU version of Colab in order to be efficient.
