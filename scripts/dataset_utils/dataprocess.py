import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# import pymorphy2
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tqdm_

import argparse
import torch

nltk.download('stopwords')
nltk.download('punkt')

def prep_to_text(sample_text):
    soup = BeautifulSoup(sample_text, 'html.parser')
    text = soup.get_text()

    text = re.sub(r'[^\w\s.,:;\-()"\'?!\\n\\r\\t\\b\\f\\v\\xa0\\u200b\\u2028\\u2029&nbsp;&amp;&lt;&gt;&quot;&apos;]',
                  ' ', text)
    text = re.sub(r'\s+', ' ', text)

    stop_words = set(stopwords.words('russian'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scrapped json.')
    parser.add_argument('--path_to_json', type=str,
                        help="Full path to not processed json file", default="news.json")
    parser.add_argument('--path_to_csv', type=str,
                        help="Full path to csv that will be saved.", default="processed_news.csv")

    parser.add_argument('--source_text_field_name', type=str, help="Name of field that includes article.", default="body")
    parser.add_argument('--annotation_field_name', type=str, help="Name of field that includes annotation", default="annotation")
    args = dict(vars(parser.parse_args()))

    assert args['path_to_json'][:-5] == '.json', f"{args['path_to_json']} is not path to json file"
    assert args['path_to_csv'][:-4] == '.csv', f"{args['path_to_csv']} is not path to json file"

    with open(args['path_to_json'], "r") as file:
        df = pd.DataFrame(json.load(file))

    df[args['source_text_field_name']].replace(['', '-', '\r\n'], np.nan, inplace=True)
    df[args['annotation_field_name']].replace(['', '-', '\r\n'], np.nan, inplace=True)
    df = df[[args['source_text_field_name'], args['annotation_field_name']]] \
        .rename(columns={args['source_text_field_name']: "document", args['annotation_field_name']: "summary"})
    df = df[df['document'].str.len() >= 100].dropna().reset_index(drop=True)

    tqdm_.pandas()
    df['document'] = df['document'].progress_apply(prep_to_text)
    df['summary'] = df['summary'].progress_apply(prep_to_text)

    df.to_csv(args['path_to_csv'], index=False)
