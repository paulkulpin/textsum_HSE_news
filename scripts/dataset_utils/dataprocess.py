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
import html

nltk.download('punkt')
nltk.download('stopwords')


def prep_doc(text: str, delete_stop_words: bool = False):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = html.unescape(text)
    #     print(text, '\n\n\n')

    text = re.sub(r'(\+7|8)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}|(\+7|8)\d{10}', r'', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', r'', text)
    text = re.sub(r'\[\d+\].*//.*', '', text)
    text = re.sub(r'.*//.*', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(
        r'(?:(?:https?://)?(?:www\.)?)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'', text)

    text = re.sub(
        r'[^\w\s.,:;()"\'?!«»“”‘’„“…=\+\*№%\[\]\-\—\–\\n\\r\\t\\b\\f\\v\\xa0\\u200b\\u2028\\u2029&nbsp;&amp;&lt;&gt;&quot;&apos;]',
        ' ', text)
    text = re.sub(r'\s+', ' ', text)

    try:
        before1, after1 = text.rsplit('.', 1)
    except:
        before1, after1 = text, '*' * 10000
    try:
        before2, after2 = text.rsplit('!', 1)
    except:
        before2, after2 = text, '*' * 10000
    try:
        before3, after3 = text.rsplit('?', 1)
    except:
        before3, after3 = text, '*' * 10000

    #     print(len(before1), len(after1), '|', len(before2), len(after2), '|', len(before3), len(after3), '|')
    if len(after1) <= len(after2) and len(after1) <= len(after3):
        text = before1 + '.'
    elif len(after2) < len(after1) and len(after2) < len(after3):
        text = before2 + '!'
    elif len(after3) < len(after1) and len(after3) < len(after2):
        text = before3 + '?'

    if text[-2:] == '..' and text[:-3] != '...':
        text = text[:-1]
    if text[-3:] == '. .' or text[-3:] == '! !' or text[-3:] == '? ?':
        text = text[:-2]

    if delete_stop_words:
        stop_words = set(stopwords.words('russian'))
        words = [word for word in text.split() if word not in stop_words]
        text = ' '.join(words)

    return text


def prep_ann(text: str, delete_stop_words: bool = False):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = html.unescape(text)
    #     print(text, '\n\n\n')

    text = re.sub(r'(\+7|8)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}|(\+7|8)\d{10}', r'', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', r'', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', r'', text)

    text = re.sub(
        r'[^\w\s.,:;()"\'?!«»“”‘’„“…=\+\*№%\[\]\-\—\–\\n\\r\\t\\b\\f\\v\\xa0\\u200b\\u2028\\u2029&nbsp;&amp;&lt;&gt;&quot;&apos;]',
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
    parser.add_argument('--delete_stop_words', type=bool, help="delete NLTK stop-words? default: False", default=False)
    parser.add_argument('--source_text_field_name', type=str, help="Name of field that includes article.", default="body")
    parser.add_argument('--annotation_field_name', type=str, help="Name of field that includes annotation", default="annotation")
    parser.add_argument('--max_document_char_length', type=int, help="Max length of document in symbols.", default=15000)
    parser.add_argument('--min_document_char_length', type=int, help="Min length of document in symbols", default=500)
    parser.add_argument('--max_summary_char_length', type=int, help="Max length of summary in symbols", default=1000)
    parser.add_argument('--min_summary_char_length', type=int, help="Min length of summary in symbols", default=20)
    parser.add_argument('--sum_doc_proportion', type=float, help="Delete all texts with sum/doc lengths proportion bigger than <value>. if 0.0, then no deleting.", default=1.5)
    
    args = dict(vars(parser.parse_args()))

    assert args['path_to_json'][-5:] == '.json', f"{args['path_to_json']} is not path to json file"
    assert args['path_to_csv'][-4:] == '.csv', f"{args['path_to_csv']} is not path to json file"

    with open(args['path_to_json'], "r") as file:
        df = pd.DataFrame(json.load(file))

    df[args['source_text_field_name']].replace(['', '-', '\r\n'], np.nan, inplace=True)
    df[args['annotation_field_name']].replace(['', '-', '\r\n'], np.nan, inplace=True)

    df = df[[args['source_text_field_name'], args['annotation_field_name']]] \
        .rename(columns={args['source_text_field_name']: "document", args['annotation_field_name']: "summary"}) \
        .dropna().reset_index(drop=True)

    df = df[df['document'].str.len() <= args['max_document_char_length']]
    df = df[df['document'].str.len() >= args['min_document_char_length']]
    df = df[df['summary'].str.len() <= args['max_summary_char_length']]
    df = df[df['summary'].str.len() >= args['min_summary_char_length']]

    if args['sum_doc_proportion'] > 0.0:
        df['ratio'] = df['summary'].str.len() / df['document'].str.len()
        df = df[df['ratio'] <= args['sum_doc_proportion']]

    tqdm_.pandas()
    df['document'] = df['document'].progress_apply(
        lambda x: prep_doc(x, delete_stop_words=args['delete_stop_words']))
    df['summary'] = df['summary'].progress_apply(
        lambda x: prep_ann(x, delete_stop_words=args['delete_stop_words']))

    df.to_csv(args['path_to_csv'], index=False)
