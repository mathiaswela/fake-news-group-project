import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def build_vectorizer(max_features: int = 10000) -> CountVectorizer:
    """Create a bag-of-words vectorizer with a fixed vocabulary size."""
    return CountVectorizer(max_features=max_features)


def build_text_features(train_text, val_text, max_features: int = 10000):
    """
    Fit CountVectorizer on train text and transform both train and validation text.
    Returns X_train, X_val, vectorizer.
    """
    vectorizer = build_vectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_text.fillna('').astype(str))
    X_val = vectorizer.transform(val_text.fillna('').astype(str))
    return X_train, X_val, vectorizer


def build_metadata_features(df) -> csr_matrix:
    """
    Simple metadata features:
    1) URL count
    2) Number count
    3) Article length in words
    """
    raw_text = df['content'].fillna('').astype(str) if 'content' in df.columns else df['content_processed'].fillna('').astype(str)
    processed_text = df['content_processed'].fillna('').astype(str) if 'content_processed' in df.columns else raw_text

    url_count = raw_text.str.count(r'https?://\S+|www\.\S+')
    number_count = raw_text.str.count(r'\b\d+\b')
    article_length = processed_text.str.count(r'\S+')

    features = np.column_stack([url_count, number_count, article_length]).astype(np.float32)
    return csr_matrix(features)


def combine_with_metadata(X_text, df) -> csr_matrix:
    """Append metadata features to a sparse text feature matrix."""
    X_meta = build_metadata_features(df)
    return hstack([X_text, X_meta], format='csr')


def train_logreg(
    X_train,
    y_train,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = 'liblinear',
    class_weight='balanced'
) -> LogisticRegression:
    """Train a simple logistic regression classifier."""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val, label_names=None) -> dict:
    """Evaluate the model and return useful metrics."""
    y_pred = model.predict(X_val)

    report = classification_report(
        y_val,
        y_pred,
        target_names=label_names,
        zero_division=0
    ) if label_names is not None else classification_report(
        y_val,
        y_pred,
        zero_division=0
    )

    return {
        'y_pred': y_pred,
        'macro_f1': f1_score(y_val, y_pred, average='macro'),
        'binary_f1': f1_score(y_val, y_pred, average='binary', zero_division=0),
        'report': report,
        'confusion_matrix': confusion_matrix(y_val, y_pred)
    }
