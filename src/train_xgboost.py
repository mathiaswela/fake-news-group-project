import gc
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack

from src.xgboost_features import (
    COLS_TO_KEEP,
    FEATURE_COLS,
    build_linguistic_sparse_matrix,
    extract_linguistic_features,
    fit_tfidf_on_training_sample,
    transform_text_in_chunks,
)


def main():
    # 1. Define paths and columns
    train_path = 'data/processed/splits/news_stratified_train.csv'
    val_path = 'data/processed/splits/news_stratified_val.csv'

    # Train data
    print("\n--- treating training data ---")
    train_df = pd.read_csv(train_path, usecols=COLS_TO_KEEP, dtype={'type': np.int8})
    train_df = extract_linguistic_features(train_df)

    # Isolate y and linguistic features
    y_train = train_df['type'].values
    X_train_ling = build_linguistic_sparse_matrix(train_df)
    tfidf = fit_tfidf_on_training_sample(train_df['content_processed'])

    X_train_text = transform_text_in_chunks(
        train_df['content_processed'],
        tfidf,
        chunk_size=50000,
        label='training',
    )

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
    val_df = pd.read_csv(val_path, usecols=COLS_TO_KEEP, dtype={'type': np.int8})
    val_df = extract_linguistic_features(val_df)

    y_val = val_df['type'].values
    X_val_ling = build_linguistic_sparse_matrix(val_df)

    print("Running chunked TF-IDF transform on validation data...")
    X_val_text = transform_text_in_chunks(
        val_df['content_processed'],
        tfidf,
        chunk_size=50000,
        label='validation',
    )

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
