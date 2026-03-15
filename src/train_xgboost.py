import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import csr_matrix, hstack


def extract_linguistic_features(df):
    """Extracting linguistic features from the 'content' column."""
    print("Calculating linguistic features...")
    df = df.copy()
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


def main():
    # 1. Define paths and columns
    train_path = 'data/processed/splits/news_time_train.csv'
    val_path = 'data/processed/splits/news_time_val.csv'
    cols_to_keep = ['title', 'content', 'content_processed', 'type']

    feature_cols = [
        'caps_ratio', 'exclamation_density', 'question_density',
        'content_word_count', 'avg_word_length', 'title_content_ratio'
    ]

    # Train data
    print("\n--- treating training data ---")
    train_df = pd.read_csv(train_path, usecols=cols_to_keep, dtype={'type': np.int8})
    train_df = extract_linguistic_features(train_df)

    # Isolate y and linguistic features
    y_train = train_df['type'].values
    X_train_ling = csr_matrix(
        train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    )

    # TF-IDF Vectorizer
    print("Running TF-IDF on training data...")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        dtype=np.float32,
        min_df=50,
        max_df=0.90,
    )
    X_train_text = tfidf.fit_transform(train_df['content_processed'].fillna(''))

    # Remove train_df for RAM purposes
    del train_df
    gc.collect()

    print("Collecting final training matrix..")
    X_train_final = hstack([X_train_text, X_train_ling], format='csr')

    del X_train_text, X_train_ling
    gc.collect()
    print(f"Training matrix ready. Shape: {X_train_final.shape}")

    # --- Val data ---
    print("\n--- Treating Validation data ---")
    val_df = pd.read_csv(val_path, usecols=cols_to_keep, dtype={'type': np.int8})
    val_df = extract_linguistic_features(val_df)

    y_val = val_df['type'].values
    X_val_ling = csr_matrix(
        val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    )

    print("Running TF-IDF transform on validation data...")
    # IMPORTANT: only .transform() here, not fit!
    X_val_text = tfidf.transform(val_df['content_processed'].fillna(''))

    del val_df
    gc.collect()

    X_val_final = hstack([X_val_text, X_val_ling], format='csr')

    del X_val_text, X_val_ling
    gc.collect()
    print(f"Validation matrix ready. Shape: {X_val_final.shape}")

    # --- MODEL TRAINING (XGBOOST) ---
    print("\n--- Fitting XGBOOST Model ---")

    # Calculate weight: negative / positive
    weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        n_jobs=8,
        scale_pos_weight=weight
    )

    xgb_model.fit(X_train_final, y_train)
    print("Training done!")

    # --- EVALUATION ---
    print("\n--- EVALUATE ON VALIDATION DATA ---")
    y_val_pred = xgb_model.predict(X_val_final)

    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Reliable (0)', 'Fake (1)']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))


if __name__ == "__main__":
    main()
