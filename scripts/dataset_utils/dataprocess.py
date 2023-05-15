import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tqdm_

import argparse

nltk.download('stopwords')
nltk.download('punkt')

def prep_to_text(sample_text, delete_stop_words=False):
    soup = BeautifulSoup(sample_text, 'html.parser')
    text = soup.get_text()

    text = re.sub(r'[^\w\s.,:;\-()"\'?!\\n\\r\\t\\b\\f\\v\\xa0\\u200b\\u2028\\u2029&nbsp;&amp;&lt;&gt;&quot;&apos;]',
                  ' ', text)
    text = re.sub(r'\s+', ' ', text)

    if delete_stop_words:
        stop_words = set(stopwords.words('russian'))
        words = [word for word in text.split() if word not in stop_words]
        text = ' '.join(words)
    
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scrapped json.')
    parser.add_argument('--path_to_json', type=str, help="Full path to not processed json file", default="news.json")
    parser.add_argument('--path_to_csv', type=str, help="Full path to csv that will be saved.", default="processed_news.csv")
    parser.add_argument('--delete_stop_words', type=bool, help="delete NLTK stop-words? True/False", default=False)
    parser.add_argument('--source_text_field_name', type=str, help="Name of field that includes article.", default="body")
    parser.add_argument('--annotation_field_name', type=str, help="Name of field that includes annotation", default="annotation")
    parser.add_argument('--max_document_char_length', type=int, help="Max length of document in symbols.", default=15000)
    parser.add_argument('--min_document_char_length', type=int, help="Min length of document in symbols", default=300)
    parser.add_argument('--max_summary_char_length', type=int, help="Max length of summary in symbols", default=1000)
    parser.add_argument('--min_summary_char_length', type=int, help="Min length of summary in symbols", default=20)
    
    args = dict(vars(parser.parse_args()))

    assert args['path_to_json'][:-5] == '.json', f"{args['path_to_json']} is not path to json file"
    assert args['path_to_csv'][:-4] == '.csv', f"{args['path_to_csv']} is not path to json file"

    with open(args['path_to_json'], "r") as file:
        df = pd.DataFrame(json.load(file))

    df[args['source_text_field_name']].replace(['', '-', '\r\n'], np.nan, inplace=True)
    df[args['annotation_field_name']].replace(['', '-', '\r\n'], np.nan, inplace=True)
    df = df[[args['source_text_field_name'], args['annotation_field_name']]] \
        .rename(columns={args['source_text_field_name']: "document", args['annotation_field_name']: "summary"})
    df = df.dropna().reset_index(drop=True)
    df = df[df['document'].str.len() <= args['max_document_char_length']]
    df = df[df['document'].str.len() >= args['min_document_char_length']]
    df = df[df['summary'].str.len() <= args['max_summary_char_length']]
    df = df[df['summary'].str.len() >= args['min_summary_char_length']]

    tqdm_.pandas()
    df['document'] = df['document'].progress_apply(lambda x: prep_to_text(x, delete_stop_words=args['delete_stop_words']))
    df['summary'] = df['summary'].progress_apply(lambda x: prep_to_text(x, delete_stop_words=args['delete_stop_words']))

    df.to_csv(args['path_to_csv'], index=False)
