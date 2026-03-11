## CSV cleaning script

The notebook flow from `notebooks/01_data_processing_mathias.ipynb` is now available as a terminal script.

Run it from the project root with:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv
```

Optional:

```bash
python -m src.clean_csv data/raw/995,000_rows.csv data/processed/995K_cleaned.csv --cores 4
```

What it does:

- loads the raw CSV
- applies the same `initial_cleaning(...)` step as notebook 01
- normalizes `content` and `title`
- tokenizes, removes stopwords, and stems into `content_processed`
- writes the cleaned CSV to the output path
