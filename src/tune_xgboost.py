import gc
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample

from src.xgboost_features import (
    COLS_TO_KEEP,
    build_linguistic_sparse_matrix,
    extract_linguistic_features,
    fit_tfidf_on_training_sample,
    transform_text_in_chunks,
)


def main():
    train_path = 'data/processed/splits/news_stratified_train.csv'

    print("\n--- treating training data for tuning ---")
    train_df = pd.read_csv(train_path, usecols=COLS_TO_KEEP, dtype={'type': np.int8})
    train_df = extract_linguistic_features(train_df)

    y_train = train_df['type'].values
    X_train_ling = build_linguistic_sparse_matrix(train_df)
    tfidf = fit_tfidf_on_training_sample(train_df['content_processed'])

    X_train_text = transform_text_in_chunks(
        train_df['content_processed'],
        tfidf,
        chunk_size=50000,
        label='training',
    )

    del train_df
    gc.collect()

    print("Collecting final training matrix...")
    X_train_final = hstack([X_train_text, X_train_ling], format='csr')

    del X_train_text, X_train_ling
    gc.collect()
    print(f"Training matrix ready. Shape: {X_train_final.shape}")

    weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    base_model = xgb.XGBClassifier(
        tree_method='hist',
        scale_pos_weight=weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=8,
    )

    param_distributions = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        scoring='f1',
        verbose=2,
        random_state=42,
        n_jobs=1,
    )

    print("\n--- Downsampling training matrix for tuning ---")
    X_tune, y_tune = resample(
        X_train_final,
        y_train,
        n_samples=150000,
        random_state=42,
        stratify=y_train,
    )

    del X_train_final, y_train
    gc.collect()

    print("\n--- Running RandomizedSearchCV ---")
    tuner.fit(X_tune, y_tune)

    print(f"Best F1 score: {tuner.best_score_:.4f}")
    print("Best parameters:")
    print(tuner.best_params_)


if __name__ == "__main__":
    main()
