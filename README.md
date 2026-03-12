# The Clinical Success Predictor

School project repository for data cleaning and preprocessing of the news dataset used in the clinical success prediction workflow.

## Project structure

```text
PythonProject/
├── data/
│   ├── raw/                     # raw input CSV files
│   └── processed/               # cleaned output CSV files
├── notebooks/
│   └── 01_data_processing_mathias.ipynb
├── src/
│   ├── preprocessing.py         # shared preprocessing functions
│   └── clean_csv.py             # terminal entry point for CSV cleaning
├── tests/
│   └── test_preprocessing.py    # pytest coverage for preprocessing
├── requirements.txt             # pinned Python dependencies
└── README.md
```

## Recommended Python version

Use Python `3.13` if possible. The checked-in virtual environment metadata was created with Python `3.13`, and that is the safest choice for matching dependencies exactly.

Python `3.12` may also work, but `3.13` is the recommended team baseline.

## First-time setup on a new machine

### 1. Clone the repository

```bash
git clone <repo-url>
cd PythonProject
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

If `python3.13` is not available, check your installed versions:

```bash
python3 --version
python3.13 --version
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install all project dependencies

```bash
pip install -r requirements.txt
```

This installs the same pinned package versions used in the project.

### 5. Install the required NLTK resources

The preprocessing code uses NLTK stopwords and tokenization resources. Run this once after setting up the environment:

```bash
python -m nltk.downloader stopwords punkt punkt_tab
```

NLTK usually stores these in `~/nltk_data`, which is now supported automatically by the code.

### 6. Verify that the environment works

Run the test suite:

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
```

If this passes, your `.venv` is set up correctly.

## Daily workflow

### Activate the environment

Each time you open a new terminal:

```bash
source .venv/bin/activate
```

### Pull the latest changes

```bash
git pull
```

### Reinstall dependencies if `requirements.txt` changed

```bash
pip install -r requirements.txt
```

### Run tests before committing

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
```

## How to clean a CSV from the terminal

The notebook flow from [notebooks/01_data_processing_mathias.ipynb](/Users/mathiaswlaursen/Desktop/MLDS fritid/the-clinical-success-predictor/PythonProject/notebooks/01_data_processing_mathias.ipynb) is also available as a script.

Run from the project root:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv
```

Optional: specify how many CPU cores to use:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv --cores 4
```

If `--cores` is not provided, the code uses:

```python
max(1, cpu_count() - 2)
```

That means it leaves 2 CPU cores free so the machine stays responsive.

## What the cleaning pipeline does

The terminal script calls `run_cleaning_pipeline(...)` in [src/preprocessing.py](/Users/mathiaswlaursen/Desktop/MLDS fritid/the-clinical-success-predictor/PythonProject/src/preprocessing.py), which:

1. loads the raw CSV with pandas
2. removes unused columns like `Unnamed: 0`, `inserted_at`, and `updated_at`
3. drops rows missing required fields such as `type`, `content`, and sometimes `domain`
4. fills missing `authors` and `title`
5. normalizes text in `content` and `title`
6. tokenizes text, removes stopwords, and stems tokens
7. writes the cleaned CSV to the requested output path

The processed text is saved in the `content_processed` column.

## How to split a dataset from the terminal

Two split modes are available, both producing `train`, `val`, and `test` CSV files in an 80/10/10 split.

Random split with `random_state=42`:

```bash
python -m src.split_data random data/processed/995K_cleaned.csv data/processed/splits --prefix news_random
```

Chronological split using `scraped_at` to avoid time leakage:

```bash
python -m src.split_data chronological data/processed/995K_cleaned.csv data/processed/splits --prefix news_time
```

If needed, you can point the chronological split to another datetime column:

```bash
python -m src.split_data chronological input.csv output_dir --date-column scraped_at
```

## Working in notebooks

Start Jupyter from the project root after activating `.venv`:

```bash
jupyter lab
```

Then open:

```text
notebooks/01_data_processing_mathias.ipynb
```

Because the notebook uses `sys.path.append('..')`, it expects to run from the repository structure as checked in.

## IDE setup

If you use PyCharm or VS Code, set the project interpreter to:

```text
PythonProject/.venv/bin/python
```

That ensures the IDE uses the same packages as the terminal.

## Git workflow for collaborators

Recommended workflow:

```bash
git checkout -b <your-branch-name>
git pull
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/test_preprocessing.py
git status
git add <files>
git commit -m "Describe your change"
git push
```

## Common issues

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root and use:

```bash
PYTHONPATH=. pytest tests/test_preprocessing.py
```

### NLTK resource errors

Install the resources again:

```bash
python -m nltk.downloader stopwords punkt punkt_tab
```

### Wrong interpreter in IDE

Make sure your IDE points to `.venv/bin/python` and not a system Python installation.

## Notes

- Do not commit `.venv/`, cache folders, or machine-specific IDE files unless they are already intentionally tracked.
- If the large raw dataset is not available on another machine, place the CSV in `data/raw/` before running the cleaning script.
