import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# In future steps, we will need to download the following packages for removing stopwords
nltk.download('stopwords') 
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Define the folder path where the HTML files are located
fpath = 'data/sec-edgar-filings'

# Create an empty list to store the data from each HTML file
disclosures = []

# Iteratively read the .txt files in the folder
for filename in os.listdir(fpath):
    if filename.endswith('.txt'):

        with open(os.path.join(fpath, filename), 'r') as file:
            # Read the text from the .txt file and append it to the list
            text = file.read()

            # Remove HTML tags from the text
            soup = BeautifulSoup(text, 'html.parser')
            text_without_tags = soup.get_text()

            # Append the text to the list
            disclosures.append({'Filename': filename, 'Text': text})

# Create a dataframe from the list of dictionaries
disclosures_df = pd.DataFrame(disclosures)

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered)

disclosures_df['clean_text'] = disclosures_df['Text'].apply(remove_stopwords)

# Extract the date from the text
disclosures_df['Date'] = disclosures_df['Text'].str.extract(r': (\d{8})', expand=False)
disclosures_df['Date'] = pd.to_datetime(disclosures_df['Date'], format='%Y%m%d')

# Extract stock ticker from the text
disclosures_df['Ticker'] = disclosures_df['Filename'].str[:4]

# For future convenience, export 'Date' and 'Ticker' to a separate dataframe
# such that realized returns will be merged to this dataframe
stocks = disclosures_df[['Ticker', 'Date']]
stocks.to_pickle('data/stocks.pkl')

# Export the dataframe to a pickle file
disclosures_df.to_pickle('data/disclosures.pkl')