# ! for Google Colab, but we will store these in environment.yml and requirements.txt
# !pip install transformers
# !pip install torch
# !pip install xformers

from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

sentences = ["there is a shortage of capital, and we need extra funding",
             "profits are flat",
             "the company is doing well, and we are making money"]

results = nlp(sentences)

# ---------------------------

tokens =tokenizer.encode_plus(txt, add_special_tokens = False, return_tensors = 'pt')
print(len(tokens))
tokens
