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
from pathlib import Path
import gc
from sklearn.model_selection import train_test_split

# NLTK setup
default_nltk_path = Path.home() / 'nltk_data'
if default_nltk_path.exists():
    nltk.data.path.append(str(default_nltk_path))
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Vocab tracking
vocab_raw = set()
vocab_no_stop = set()
vocab_stemmed = set()


def reset_vocab_tracking():
    vocab_raw.clear()
    vocab_no_stop.clear()
    vocab_stemmed.clear()

def parallel_process(df, func, n_cores=None):
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 2)

    # Use pandas iloc to split safely and prevent conversion to numpy arrays in modern NumPy/Pandas versions
    chunk_size = max(1, int(np.ceil(len(df) / n_cores)))
    df_split = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with mp.Pool(n_cores) as pool:
        results = pool.map(func, df_split)

    # If the function returns a tuple, assume it's (df, vocab_raw, vocab_no_stop, vocab_stemmed)
    if isinstance(results[0], tuple):
        df_res = pd.concat([r[0] for r in results])
        
        global vocab_raw, vocab_no_stop, vocab_stemmed
        for r in results:
            vocab_raw.update(r[1])
            vocab_no_stop.update(r[2])
            vocab_stemmed.update(r[3])
            
        return df_res
    else:
        return pd.concat(results)

# Wrappers for parallel processing
def wrapper_normalize(df_subset):
    # Make a copy to prevent SettingWithCopyWarning
    df_subset = df_subset.copy()
    df_subset['content_normalized'] = df_subset['content'].apply(normalize_text)
    if 'title' in df_subset.columns:
        df_subset['title_normalized'] = df_subset['title'].fillna("").apply(normalize_text)
    return df_subset

def wrapper_tokenize(df_subset):
    # Create LOCAL sets in each worker instead of using global ones
    local_raw = set()
    local_no_stop = set()
    local_stemmed = set()

    def internal_logic(text):
        # Faster split instead of word_tokenize, since punctuation IS gone
        tokens = text.split() 
        local_raw.update(tokens)
        
        no_stop = [w for w in tokens if w not in stop_words]
        local_no_stop.update(no_stop)
        
        stemmed = [stemmer.stem(w) for w in no_stop]
        local_stemmed.update(stemmed)
        
        return " ".join(stemmed)

    # Make a copy to prevent SettingWithCopyWarning
    df_subset = df_subset.copy()
    df_subset['content_processed'] = df_subset['content_normalized'].apply(internal_logic)
    
    # Return everything to the main process
    return df_subset, local_raw, local_no_stop, local_stemmed

def ensure_directories():
    paths = ['../data/raw', '../data/processed', '../notebooks']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

def initial_cleaning(df):
    initial_rows = len(df)

    df_clean = df.copy()

    cols_to_drop = ['Unnamed: 0', 'inserted_at', 'updated_at']
    protected_columns = {'id', 'type', 'content', 'domain', 'authors', 'title', 'scraped_at'}
    for col in df_clean.columns:
        if col not in protected_columns and df_clean[col].isnull().all():
            cols_to_drop.append(col)

    df_clean = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])
    
    # Drop rows with missing identifiers and required fields
    subset_to_drop = ['id', 'type', 'content']
    if 'domain' in df_clean.columns:
        subset_to_drop.append('domain')
    df_clean = df_clean.dropna(subset=subset_to_drop)

    # Fill missing values for authors and title
    if 'authors' in df_clean.columns:
        df_clean['authors'] = df_clean['authors'].fillna('unknown_author')
    
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].fillna('no_title')

    df_clean['content'] = df_clean['content'].astype(str)
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].astype(str)

    if 'scraped_at' in df_clean.columns:
        # Standardize string formats by replacing 'T' with a space
        # Use format='mixed' to cleanly parse varying decimal seconds and lengths
        # Use utc=True to handle mixed timezones without throwing a ValueError
        df_clean['scraped_at'] = df_clean['scraped_at'].astype(str).str.replace('T', ' ')
        df_clean['scraped_at'] = pd.to_datetime(df_clean['scraped_at'], format='mixed', errors='coerce', utc=True)
        missing_scraped_at = df_clean['scraped_at'].isna().sum()
        if missing_scraped_at:
            df_clean = df_clean.dropna(subset=['scraped_at'])
        print(f"Dropped {missing_scraped_at} rows with missing or invalid scraped_at.")

    df_clean['type'] = np.where(
        df_clean['type'].isin(['reliable', 'political']),
        0,
        1,
    )

    final_rows = len(df_clean)
    print(f"Cleaning done: Start={initial_rows}, End={final_rows}, Dropped={initial_rows - final_rows} rows.")

    return df_clean

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    # Replace URLs manually BEFORE replacing slashes (otherwise slashes are lost and URL isn't recognized)
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, 'URLTOKEN', text)

    date_pattern = r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b'
    text = re.sub(date_pattern, 'DATETOKEN', text)

    # Now it is safe to replace dashes and slashes
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


def run_cleaning_pipeline(
    input_path,
    output_path,
    n_cores=None,
    split_output_dir=None,
    split_prefix=None,
    split_method='chronological',
    chunksize=100000,
    print_summary=True,
):
    reset_vocab_tracking()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if split_output_dir is None:
        split_output_dir = output_dir or '.'
    os.makedirs(split_output_dir, exist_ok=True)

    if split_prefix is None:
        split_prefix = Path(output_path).stem

    if os.path.exists(output_path):
        os.remove(output_path)

    total_rows_read = 0
    total_rows_written = 0
    chunks_written = 0
    has_written_output = False
    chunk_index = 0
    output_columns = None

    chunks = pd.read_csv(
        input_path,
        low_memory=False,
        dtype={'Unnamed: 0': str, 'id': str},
        chunksize=chunksize,
    )

    for chunk_index, chunk in enumerate(chunks, start=1):
        total_rows_read += len(chunk)

        if print_summary:
            print(f"Processing chunk {chunk_index} with {len(chunk):,} rows")

        chunk = initial_cleaning(chunk)
        if chunk.empty:
            if print_summary:
                print(f"Chunk {chunk_index} is empty after cleaning; skipping write.")
            del chunk
            gc.collect()
            continue

        if print_summary:
            print(f"Chunk {chunk_index}: normalizing content and title")
        chunk = parallel_process(chunk, wrapper_normalize, n_cores=n_cores)

        if print_summary:
            print(f"Chunk {chunk_index}: tokenizing, removing stopwords, and stemming")
        chunk = parallel_process(chunk, wrapper_tokenize, n_cores=n_cores)

        if output_columns is None:
            output_columns = list(chunk.columns)
        else:
            chunk = chunk.reindex(columns=output_columns)

        chunk.to_csv(
            output_path,
            mode='w' if not has_written_output else 'a',
            header=not has_written_output,
            index=False,
        )

        has_written_output = True
        chunks_written += 1
        total_rows_written += len(chunk)

        del chunk
        gc.collect()

    if not has_written_output:
        raise ValueError("No rows were written to the cleaned output file.")

    processed_df = pd.read_csv(output_path, low_memory=False, dtype={'id': str})
    if split_method == 'chronological':
        train_df, val_df, test_df = chronological_split_dataframe(processed_df)
    elif split_method == 'stratified':
        train_df, val_df, test_df = stratified_split_dataframe(processed_df)
    else:
        raise ValueError(f"Unsupported split_method: {split_method}")

    train_path, val_path, test_path = save_split_dataframes(
        train_df,
        val_df,
        test_df,
        split_output_dir,
        split_prefix,
    )

    if print_summary:
        print(f"Processed {total_rows_read:,} input rows across {chunk_index} chunk(s)")
        print(f"Wrote {total_rows_written:,} cleaned rows across {chunks_written} chunk(s)")
        print(f"Saved cleaned CSV to {output_path}")
        print(f"Saved {split_method} train split to {train_path}")
        print(f"Saved {split_method} validation split to {val_path}")
        print(f"Saved {split_method} test split to {test_path}")
        print_reduction_rates()

    return {
        'processed_path': output_path,
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,
    }


def _calculate_split_boundaries(n_rows, train_frac=0.8, val_frac=0.1):
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    return train_end, val_end


def random_split_dataframe(df, random_state=42):
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_end, val_end = _calculate_split_boundaries(len(shuffled_df))

    train_df = shuffled_df.iloc[:train_end].copy()
    val_df = shuffled_df.iloc[train_end:val_end].copy()
    test_df = shuffled_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def chronological_split_dataframe(df, date_column='scraped_at'):
    if date_column not in df.columns:
        raise ValueError(f"Missing required date column: {date_column}")
    if 'id' not in df.columns:
        raise ValueError("Missing required tie-breaker column: id")

    working_df = df.copy()
    working_df[date_column] = pd.to_datetime(working_df[date_column], format='mixed', errors='coerce', utc=True)

    if working_df[date_column].isna().any():
        raise ValueError(f"Column '{date_column}' contains invalid or missing datetimes")

    ordered_df = working_df.sort_values([date_column, 'id']).reset_index(drop=True)
    train_end, val_end = _calculate_split_boundaries(len(ordered_df))

    train_df = ordered_df.iloc[:train_end].copy()
    val_df = ordered_df.iloc[train_end:val_end].copy()
    test_df = ordered_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def stratified_split_dataframe(df, label_column='type', random_state=42):
    if label_column not in df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[label_column],
        random_state=random_state,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[label_column],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split_dataframes(train_df, val_df, test_df, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"{prefix}_train.csv")
    val_path = os.path.join(output_dir, f"{prefix}_val.csv")
    test_path = os.path.join(output_dir, f"{prefix}_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def run_random_split(input_path, output_dir, prefix='random_split'):
    df = pd.read_csv(input_path, low_memory=False)
    train_df, val_df, test_df = random_split_dataframe(df, random_state=42)
    return save_split_dataframes(train_df, val_df, test_df, output_dir, prefix)


def run_chronological_split(input_path, output_dir, prefix='chronological_split', date_column='scraped_at'):
    df = pd.read_csv(input_path, low_memory=False)
    train_df, val_df, test_df = chronological_split_dataframe(df, date_column=date_column)
    return save_split_dataframes(train_df, val_df, test_df, output_dir, prefix)


def run_stratified_split(input_path, output_dir, prefix='stratified_split', label_column='type'):
    df = pd.read_csv(input_path, low_memory=False)
    train_df, val_df, test_df = stratified_split_dataframe(df, label_column=label_column)
    return save_split_dataframes(train_df, val_df, test_df, output_dir, prefix)
