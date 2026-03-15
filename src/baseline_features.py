import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix


def build_count_vectorizer(max_features: int = 10000) -> CountVectorizer:
    """Return a CountVectorizer with top max_features words."""
    return CountVectorizer(max_features=max_features)


def extract_metadata_features(df) -> csr_matrix:
    """Extract URL count, number count and article length from raw content."""
    url_pattern = r'https?://\S+|www\.\S+'
    num_pattern = r'\b\d+\b'

    url_counts = df['content'].astype(str).apply(lambda x: len(re.findall(url_pattern, x)))
    num_counts = df['content'].astype(str).apply(lambda x: len(re.findall(num_pattern, x)))
    content_length = df['content_processed'].astype(str).apply(lambda x: len(x.split()))

    features = np.column_stack([url_counts, num_counts, content_length])
    return csr_matrix(features.astype(np.float32))


def build_feature_matrix(df, vectorizer, fit: bool = False) -> csr_matrix:
    """Combine CountVectorizer text features and metadata features."""
    if fit:
        text_features = vectorizer.fit_transform(df['content_processed'].fillna(''))
    else:
        text_features = vectorizer.transform(df['content_processed'].fillna(''))

    meta_features = extract_metadata_features(df)
    return hstack([text_features, meta_features], format='csr')