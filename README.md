# Fake News Detection Project

This repository contains the code used for our course project on fake news detection. The project covers:

- data cleaning and preprocessing of FakeNewsCorpus samples
- train/validation/test splitting
- a baseline model workflow
- an advanced XGBoost text-classification workflow
- evaluation notebooks for the FakeNewsCorpus and LIAR datasets

This README is written as a reproducibility guide.
## Repository structure

```text
PythonProject/
├── data/
│   ├── raw/                          # raw CSV files
│   └── processed/                    # cleaned CSVs and split CSVs
├── models/                           # saved TF-IDF vectorizer and trained XGBoost model
├── notebooks/
│   ├── 01_data_processing_mathias.ipynb
│   ├── 02_data_exploration_andreas.ipynb
│   ├── 03_baseline_model_andreas.ipynb
│   ├── 04_evaluation_andreas.ipynb
│   ├── 05_xgboost_model_evalutation_LIAR_mathias.ipynb
│   └── 06_xgboost_model_evaluation_mathias.ipynb
├── src/
│   ├── baseline_features.py
│   ├── clean_csv.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── train_xgboost.py
│   ├── tune_xgboost.py
│   └── xgboost_features.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_xgboost_pipeline.py
├── requirements.txt
└── README.md
```

## Recommended Python version

Use Python `3.13` if possible.

## Setup on a new machine

### 1. Clone the repository

```bash
git clone <repo-url>
cd PythonProject
```

### 2. Create and activate a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install NLTK resources

The preprocessing code requires NLTK stopwords and tokenization resources:

```bash
python -m nltk.downloader stopwords punkt punkt_tab
```

## Quick verification

To test if the virtual environment you can run the pytests:

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
PYTHONPATH=. pytest tests/test_xgboost_pipeline.py
```

If both pass, the environment is ready.

## Data required

The code expects the following data files or folders to be available locally:

- `data/raw/news_sample.csv`
- `data/raw/995,000_rows.csv`
- `data/liar-data/liar_dataset/train.tsv`
- `data/liar-data/liar_dataset/valid.tsv`
- `data/liar-data/liar_dataset/test.tsv`

The large processed files and saved models may not be tracked in Git because of size. If they are missing, recreate them using the commands below.

## Reproducing the project results

The recommended order is:

1. preprocess the FakeNewsCorpus data
2. create the data splits
3. train the final XGBoost or baseline Logistic Regression model
4. optionally run hyperparameter tuning for XBGoost model
5. open the notebooks for analysis and figures

### A. Preprocess the large FakeNewsCorpus subset

This uses the chunked cleaning pipeline in [src/clean_csv.py] and [src/preprocessing.py]
Example command:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv --cores 4 --split-method stratified --split-output-dir data/processed/splits --split-prefix news_stratified
```

Expected split files:

- `data/processed/splits/news_stratified_train.csv`
- `data/processed/splits/news_stratified_val.csv`
- `data/processed/splits/news_stratified_test.csv`

### B. Optional: run split logic separately

If you already have the cleaned CSV and only want the split files:

```bash
python -m src.split_data stratified data/processed/995K_cleaned.csv data/processed/splits --prefix news_stratified
```


### C. Train the final XGBoost model

The final text-only XGBoost pipeline is implemented in [src/train_xgboost.py], with shared TF-IDF logic in [src/xgboost_features.py]
Run:

```bash
python -m src.train_xgboost
```

Saved model artifacts:

- `models/tfidf_vectorizer300.joblib`
- `models/xgboost_model300.json`

### D. Optional: rerun hyperparameter tuning

Hyperparameter tuning is implemented in [src/tune_xgboost.py]

```bash
python -m src.tune_xgboost
```

### E. Open the notebooks

Start Jupyter Lab from the project root:

```bash
jupyter lab
```

## Implementation details for reproducibility

### Preprocessing

The preprocessing pipeline:

- removes irrelevant columns
- drops rows with missing required data
- converts `scraped_at` to datetime
- drops rows with invalid `scraped_at`
- binarizes labels:
  - `reliable` and `political` -> `0`
  - all others -> `1`
- keeps raw text while also creating processed text columns

### XGBoost text representation

The final XGBoost model is text-only and does not use the 6 linguistic metadata features anymore.

TF-IDF settings are defined in [src/xgboost_features.py]

- `ngram_range=(1, 2)`
- `max_features=1500`
- `stop_words='english'`
- `min_df=10`
- `max_df=0.90`
- `dtype=np.float32`

The vocabulary is fit on a sample of up to `300000` rows from the training split and then applied to the full data in chunks of `50000` rows.

### Final XGBoost parameters

The current training script uses:

- `n_estimators=400`
- `max_depth=12`
- `learning_rate=0.2`
- `subsample=1.0`
- `colsample_bytree=0.9`
- `tree_method='hist'`
- `random_state=42`
- `eval_metric='logloss'`
- `n_jobs=8`
- `scale_pos_weight` computed from the training labels


```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv --cores 4 --split-method stratified --split-output-dir data/processed/splits --split-prefix news_stratified
python -m src.train_xgboost
jupyter lab
```
