# The Clinical Success Predictor

Collaborative school project for preprocessing a large news dataset and preparing train/validation/test splits for downstream machine learning.

## Project purpose

The preprocessing pipeline is built to support two ML workflows:

- text classification with a binary target derived from the `type` column
- feature engineering on preserved raw text alongside normalized and stemmed text

## Environment setup

Use Python `3.13` if possible.

From the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab
```

Verify the environment:

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
```

## Daily workflow

Activate the virtual environment before working:

```bash
source .venv/bin/activate
```

If dependencies changed:

```bash
pip install -r requirements.txt
```

Run tests before committing:

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
```

## Cleaning pipeline

The cleaning pipeline is designed for large CSV files and runs in chunks to reduce RAM usage.

Expected behavior:

- reads the input CSV in chunks of `100000` rows
- drops `Unnamed: 0`
- keeps the real `id` column
- drops rows with missing `id`
- converts `scraped_at` to datetime
- drops rows with invalid or missing `scraped_at`
- binarizes `type`
  - `reliable` and `political` -> `0`
  - everything else -> `1`
- preserves raw `content`
- creates `content_normalized`
- creates `title_normalized`
- creates `content_processed`
- appends cleaned chunks to disk instead of holding the full file in memory
- after the final cleaned CSV is written, reads it back in and creates a chronological `80/10/10` split by `scraped_at` and `id`

### Run the full cleaning pipeline

From the project root:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv
```

Run with 4 CPU cores:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv --cores 4
```

Write the chronological split files to a specific directory:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv --cores 4 --split-output-dir data/processed/splits --split-prefix news_time
```

This produces:

- the cleaned full CSV at the output path you pass in
- a chronological train split
- a chronological validation split
- a chronological test split

With the example above, the split files will be:

- `data/processed/splits/news_time_train.csv`
- `data/processed/splits/news_time_val.csv`
- `data/processed/splits/news_time_test.csv`

### Stop a running job

In the terminal where the script is running:

```bash
Ctrl+C
```

If needed, find and stop the Python process:

```bash
ps aux | grep python
kill <PID>
```

## Splitting an already cleaned dataset

If you already have a cleaned CSV and just want the split files, use the split command directly.

### Random split

Uses `80/10/10` with `random_state=42`.

```bash
python -m src.split_data random data/processed/995K_cleaned.csv data/processed/splits --prefix news_random
```

### Chronological split

Uses `80/10/10`, sorted by `scraped_at` and then `id` to avoid leakage and make the split reproducible.

```bash
python -m src.split_data chronological data/processed/995K_cleaned.csv data/processed/splits --prefix news_time
```

This requires both of these columns to exist:

- `scraped_at`
- `id`

## Output columns

After cleaning, the main text-related columns are:

- `content`: original raw text
- `content_normalized`: cleaned and normalized text
- `content_processed`: tokenized, stopword-removed, stemmed text
- `title`: original title
- `title_normalized`: normalized title
- `type`: binary label

## Troubleshooting

### `Column 'scraped_at' contains invalid or missing datetimes`

That means the cleaned file still contains bad datetime values, or you are trying to split a file that was not cleaned with the current pipeline.

Re-run the cleaning pipeline so invalid `scraped_at` rows are dropped automatically.

### `Missing required tie-breaker column: id`

The chronological split needs a real `id` column. It does not use the pandas index.

### `No module named 'src'`

Run commands from the project root and use:

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
```

## Git notes

- `.venv/`, `__pycache__/`, `.pytest_cache/`, and local IDE files should not be committed
- if tracked cache files show up in git, remove them from the index with:

```bash
git rm --cached tests/__pycache__/*.pyc
```

- review `git status` before committing
