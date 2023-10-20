import os
import json
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# nltk.download("punkt")
# nltk.download("stopwords")

oas_files = os.listdir('apis.guru-data')
print(oas_files)
print('Number of files: ', len(oas_files))

# if number of files is < 3993, then download the files from the links in apis.guru-data/urls
# if len(oas_files) < 3993:
#     print("Downloading files...")
#     os.system('wget -i apis.guru-data/urls -P apis.guru-data/')
#     print("Download complete!")


useless_words = ['', 'api', 'apis', 'rest', 'http', 'https', 'www',
                 'com', 'org', 'net', 'io', 'api', 'apis', 'rest', 'ok', 'success',
                 'error', 'errors', 'code', 'codes', 'message', 'messages', 'status',
                 'parameter', 'parameters', 'request', 'response', 'responses', 'request',
                 'requests', 'example', 'examples', 'parameter', 'parameters', 'value',
                 'values', 'object', 'objects', 'array', 'arrays', 'field', 'fields']


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)


train_data = list()
files = os.listdir('apis.guru-data')

for file in files:
    print(files.index(file))
    with (open('apis.guru-data/' + file, 'r') as f):
        try:
            data = json.load(f)
            df = pd.json_normalize(data)
        except Exception as e:
            print(f"Error in {file} : {e}")
            continue

        try:
            desc_cols = [col for col in df.columns if 'description' in col]
            desc_val = df[desc_cols].values.tolist()[0]
        except Exception as e:
            print(f"Error when extracting description in {file} : {e}")
            continue

        try:
            desc_val = " ".join(list(filter(lambda x: x not in useless_words, list(map(lambda x: preprocess_text(x), list(set(desc_val)))))))
            categories = df["info.x-apisguru-categories"].values.tolist()[
                0] if "info.x-apisguru-categories" in df.columns else []
            for category in categories:
                train_data.append([category, desc_val])
        except Exception as e:
            print(f"Error when extracting categories in {file} : {e}")
            continue

print(train_data)

train_data_df = pd.DataFrame(train_data, columns=['category', 'text'])
train_data_df.to_csv('train_data.csv', index=False)

