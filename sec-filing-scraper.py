from sec_edgar_downloader import Downloader
import os

# Create the Downloader object with relative output directory path for reproducability
current_dir = os.getcwd() 
dl = Downloader(os.path.join(current_dir, 'data')) 

# list of sampled tickers
tickers = ["NFLX", "MSFT"]

# download all 10-Ks submitted between 2017-01-01 and 2017-03-25
for ticker in tickers:
    dl.get("10-K", ticker, after="2017-01-01", before="2019-01-01")
