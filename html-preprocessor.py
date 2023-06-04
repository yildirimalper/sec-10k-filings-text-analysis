import os
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

# Define the folder path where the HTML files are located
cwd = os.getcwd()
folder_path = os.path.abspath(os.path.join(cwd, 'data/sec-edgar-filings'))

# Create an empty list to store the data from each HTML file
text_data = []

# Loop through all the HTML files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.html'):
        # Read the HTML file and extract the relevant data
        with open(os.path.join(folder_path, filename), 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Extract the relevant data from the HTML file and append it to the list
            # For example, you can extract the text from all the <p> tags
            text = ''.join([p.get_text() for p in soup.find_all('p')])
            text_data.append({'filename': filename, 'text': text})

# Create a dataframe from the list of dictionaries
text_df = pd.DataFrame(text_data)

# text_df.to_csv("text_data.csv")