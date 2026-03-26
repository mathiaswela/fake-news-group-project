import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.xgboost_features import (
    fit_tfidf_on_training_sample,
    transform_text_in_chunks,
)


def main():
    # 1. Define paths and columns
    train_path = 'data/processed/splits/news_stratified_train.csv'
    val_path = 'data/processed/splits/news_stratified_val.csv'
    cols_to_keep = ['content_processed', 'type']
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    tfidf_path = models_dir / 'tfidf_vectorizer1500ot.joblib'
    xgb_model_path = models_dir / 'xgboost_model1500ot.json'

    # Train data
    print("\n--- treating training data ---")
    train_df = pd.read_csv(train_path, usecols=cols_to_keep, dtype={'type': np.int8})

    y_train = train_df['type'].values
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

    X_train_final = X_train_text
    del X_train_text
    print(f"Training matrix ready. Shape: {X_train_final.shape}")

    # --- Val data ---
    print("\n--- Treating Validation data ---")
    val_df = pd.read_csv(val_path, usecols=cols_to_keep, dtype={'type': np.int8})

    y_val = val_df['type'].values

    print("Running chunked TF-IDF transform on validation data...")
    X_val_text = transform_text_in_chunks(
        val_df['content_processed'],
        tfidf,
        chunk_size=50000,
        label='validation',
    )

    del val_df
    gc.collect()

    X_val_final = X_val_text
    del X_val_text
    print(f"Validation matrix ready. Shape: {X_val_final.shape}")

    # --- MODEL TRAINING (XGBOOST) ---
    print("\n--- Fitting XGBOOST Model ---")

    # Calculate weight: negative / positive
    weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=12,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=0.9,
        tree_method='hist',
        random_state=42,
        eval_metric='logloss',
        n_jobs=8,
        scale_pos_weight=weight
    )

    xgb_model.fit(X_train_final, y_train)
    print("Training done!")

    joblib.dump(tfidf, tfidf_path)
    xgb_model.save_model(xgb_model_path)
    print(f"Saved TF-IDF vectorizer to {tfidf_path}")
    print(f"Saved XGBoost model to {xgb_model_path}")

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
