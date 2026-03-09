import pandas as pd
import re
from cleantext import clean
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import os
import multiprocessing as mp
import numpy as np

# NLTK setup
nltk.data.path.append('/Users/mathiaswlaursen/nltk_data')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Vocab tracking
vocab_raw = set()
vocab_no_stop = set()
vocab_stemmed = set()

def parallel_process(df, func, n_cores=None):
    if n_cores is None:
        n_cores = mp.cpu_count() - 2

    df_split = np.array_split(df, n_cores)

    with mp.Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))

    return df

# Wrappers for parralel process
def wrapper_normalize(df_subset):
    df_subset['content'] = df_subset['content'].apply(normalize_text)
    if 'title' in df_subset.columns:
        df_subset['title'] = df_subset['title'].fillna("").apply(normalize_text)
    return df_subset

def wrapper_tokenize(df_subset):
    df_subset['content'] = df_subset['content'].apply(process_and_tokenize)
    return df_subset

def ensure_directories():
    paths = ['../data/raw', '../data/processed', '../notebooks']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

def initial_cleaning(df):
    initial_rows = len(df)

    cols_to_drop = ['Unnamed: 0', 'inserted_at', 'updated_at']
    for col in df.columns:
        if df[col].isnull().all():
            cols_to_drop.append(col)

    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df_clean = df_clean.dropna(subset=['type', 'content'])

    df_clean['content'] = df_clean['content'].astype(str)
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].astype(str)

    if 'scraped_at' in df_clean.columns:
        df_clean['scraped_at'] = pd.to_datetime(df_clean['scraped_at'], errors='coerce')

    final_rows = len(df_clean)
    print(f"Cleaning done: Start={initial_rows}, End={final_rows}, Dropped={initial_rows - final_rows} rows.")

    return df_clean

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    date_pattern = r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b'
    text = re.sub(date_pattern, 'DATETOKEN', text)

    text = text.replace("-", " ").replace("/", " ")

    cleaned_text = clean(text,
                         lower=True,
                         no_line_breaks=True,
                         no_urls=True,
                         replace_with_url="URLTOKEN",
                         no_emails=True,
                         replace_with_email="EMAILTOKEN",
                         no_numbers=True,
                         replace_with_number="NUMTOKEN",
                         no_punct=True,
                         )

    return cleaned_text

def process_and_tokenize(text):
    global vocab_raw, vocab_no_stop, vocab_stemmed

    tokens = word_tokenize(text)
    vocab_raw.update(tokens)

    tokens_no_stop = [w for w in tokens if w not in stop_words]
    vocab_no_stop.update(tokens_no_stop)

    tokens_stemmed = [stemmer.stem(w) for w in tokens_no_stop]
    vocab_stemmed.update(tokens_stemmed)

    return " ".join(tokens_stemmed)

def print_reduction_rates():
    v_raw = len(vocab_raw)
    v_stop = len(vocab_no_stop)
    v_stem = len(vocab_stemmed)

    reduction_stop = ((v_raw - v_stop) / v_raw) * 100 if v_raw > 0 else 0
    reduction_stem = ((v_stop - v_stem) / v_stop) * 100 if v_stop > 0 else 0
    total_reduction = ((v_raw - v_stem) / v_raw) * 100 if v_raw > 0 else 0

    print("\n--- Final reduction rates ---")
    print(f"Original vocabulary: {v_raw:,} unique tokens")
    print(f"After stopwords: {v_stop:,} tokens ({reduction_stop:.2f}% reduction)")
    print(f"After stemming: {v_stem:,} tokens ({reduction_stem:.2f}% reduction from previous)")
    print(f"Total reduction: {total_reduction:.2f}%")