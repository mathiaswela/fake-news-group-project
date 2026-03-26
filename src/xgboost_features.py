import gc
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_COLS = [
    'caps_ratio', 'exclamation_density', 'question_density',
    'content_word_count', 'avg_word_length', 'title_content_ratio'
]

COLS_TO_KEEP = ['title', 'content', 'content_processed', 'type']


def extract_linguistic_features(df):
    """Extracting linguistic features from the 'content' column."""
    print("Calculating linguistic features...")
    df['content'] = df['content'].astype(str)
    df['title'] = df['title'].fillna("").astype(str)

    df['caps_ratio'] = df['content'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['exclamation_density'] = df['content'].apply(lambda x: x.count('!') / (len(x) + 1))
    df['question_density'] = df['content'].apply(lambda x: x.count('?') / (len(x) + 1))
    df['content_word_count'] = df['content'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['content'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    df['title_content_ratio'] = df['title'].apply(lambda x: len(x)) / (df['content'].apply(lambda x: len(x)) + 1)

    return df


def build_linguistic_sparse_matrix(df):
    return csr_matrix(df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False))


def fit_tfidf_on_training_sample(train_series):
    print("Running memory-efficient TF-IDF on training data...")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1500,
        stop_words='english',
        min_df=10,
        max_df=0.90,
        dtype=np.float32,
    )

    sample_size = min(300000, len(train_series))
    tfidf_sample = train_series.fillna('').sample(
        n=sample_size,
        random_state=42,
    )
    tfidf.fit(tfidf_sample)

    del tfidf_sample
    gc.collect()

    return tfidf


def transform_text_in_chunks(series, tfidf, chunk_size=50000, label="dataset"):
    sparse_chunks = []

    for start in range(0, len(series), chunk_size):
        end = min(start + chunk_size, len(series))
        print(f"TF-IDF transform on {label} rows {start:,} to {end:,}...")

        text_chunk = series.iloc[start:end].fillna('')
        sparse_chunk = tfidf.transform(text_chunk)
        sparse_chunks.append(sparse_chunk)

        del text_chunk
        del sparse_chunk
        gc.collect()

    combined = vstack(sparse_chunks, format='csr')

    del sparse_chunks
    gc.collect()

    return combined
