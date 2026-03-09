import pandas as pd
import re
from cleantext import clean
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# NLTK setup
nltk.data.path.append('/Users/mathiaswlaursen/nltk_data')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Vocab tracking
vocab_raw = set()
vocab_no_stop = set()
vocab_stemmed = set()


def initial_cleaning(df):
    """
    Initial cleaning of the dataset, including dropping unnecessary columns and rows,.
    """
    initial_rows = len(df)

    # Fremove index, unnecessary date colums and columns with 0 entries
    cols_to_drop = ['Unnamed: 0', 'inserted_at', 'updated_at']
    for col in df.columns:
        if df[col].isnull().all():
            cols_to_drop.append(col)

    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Remove rows with missing 'type' (target) and 'content' main feature
    df_clean = df_clean.dropna(subset=['type', 'content'])

    # Handle datatypes
    df_clean['content'] = df_clean['content'].astype(str)
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].astype(str)

    if 'scraped_at' in df_clean.columns:
        df_clean['scraped_at'] = pd.to_datetime(df_clean['scraped_at'], errors='coerce')

    final_rows = len(df_clean)
    print(f"Cleaning done: Start={initial_rows}, End={final_rows}, Dropped={initial_rows - final_rows} rows.")

    return df_clean


def normalize_text(text):
    """
    Normalize text by removing URLs, emails, numbers, punctuation, and other unwanted characters.
    """
    if not isinstance(text, str):
        return ""

    # Handle dates before text
    date_pattern = r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b'
    text = re.sub(date_pattern, 'DATETOKEN', text)

    # Fix words that melt together during removal of punctuation
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
    """
    Tokenize, remove stopwords, and stem the text.
    """
    global vocab_raw, vocab_no_stop, vocab_stemmed

    # 1. Tokenize
    tokens = word_tokenize(text)
    vocab_raw.update(tokens)

    # 2. Remove stopwords
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    vocab_no_stop.update(tokens_no_stop)

    # 3. Stemming
    tokens_stemmed = [stemmer.stem(w) for w in tokens_no_stop]
    vocab_stemmed.update(tokens_stemmed)

    return " ".join(tokens_stemmed)


def print_reduction_rates():
    """
    Print and return the reduction rates for the different stages of the preprocessing pipeline.
    """
    v_raw = len(vocab_raw)
    v_stop = len(vocab_no_stop)
    v_stem = len(vocab_stemmed)

    reduction_stop = ((v_raw - v_stop) / v_raw) * 100 if v_raw > 0 else 0
    reduction_stem = ((v_stop - v_stem) / v_stop) * 100 if v_stop > 0 else 0
    total_reduction = ((v_raw - v_stem) / v_raw) * 100 if v_raw > 0 else 0

    print("\n--- Endelige Reduktionsrater ---")
    print(f"Originalt vokabular: {v_raw:,} unikke tokens")
    print(f"Efter stopwords: {v_stop:,} tokens ({reduction_stop:.2f}% reduktion)")
    print(f"Efter stemming: {v_stem:,} tokens ({reduction_stem:.2f}% reduktion fra forrige)")
    print(f"Total reduktion: {total_reduction:.2f}%")