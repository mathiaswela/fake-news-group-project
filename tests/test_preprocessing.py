import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import (
    chronological_split_dataframe,
    initial_cleaning,
    normalize_text,
    random_split_dataframe,
    wrapper_normalize,
    wrapper_tokenize,
    process_and_tokenize,
    parallel_process,
    run_cleaning_pipeline,
)


def dummy_func(df_subset):
    df_subset = df_subset.copy()
    df_subset['val'] = df_subset['val'] * 2
    return df_subset

def dummy_func_tuple(df_subset):
    df_subset = df_subset.copy()
    df_subset['val'] = df_subset['val'] * 2
    # Return df, raw, no_stop, stemmed
    return df_subset, {'raw1'}, {'nostop1'}, {'stem1'}

@pytest.fixture
def sample_unclean_df():
    return pd.DataFrame({
        'Unnamed: 0': [0, 1, 2, 3],
        'id': ['10', np.nan, '30', '40'],
        'inserted_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'updated_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'type': ['reliable', np.nan, 'blog', 'news'],
        'content': ['Some text here.', 'More text.', np.nan, 'Valid content.'],
        'domain': ['example.com', 'test.com', 'demo.com', np.nan],
        'authors': ['Alice', np.nan, 'Charlie', np.nan],
        'title': [np.nan, 'Title B', 'Title C', 'Title D'],
        'scraped_at': ['2023-10-01', 'invalid_date', '2023-10-03', '2023-10-04'],
        'all_null': [np.nan, np.nan, np.nan, np.nan]
    })

def test_initial_cleaning(sample_unclean_df):
    df_clean = initial_cleaning(sample_unclean_df)
    
    assert 'Unnamed: 0' not in df_clean.columns
    assert 'id' in df_clean.columns
    assert 'inserted_at' not in df_clean.columns
    assert 'updated_at' not in df_clean.columns
    assert 'all_null' not in df_clean.columns

    assert len(df_clean) == 1
    assert df_clean.iloc[0]['id'] == '10'
    assert df_clean.iloc[0]['type'] == 0
    assert df_clean.iloc[0]['title'] == 'no_title'
    assert df_clean.iloc[0]['authors'] == 'Alice'
    assert isinstance(df_clean.iloc[0]['content'], str)
    assert isinstance(df_clean.iloc[0]['title'], str)
    assert pd.api.types.is_datetime64_any_dtype(df_clean['scraped_at'])
    assert pd.api.types.is_integer_dtype(df_clean['type'])

def test_normalize_text():
    assert normalize_text(None) == ""
    assert normalize_text(np.nan) == ""

    assert "datetoken" in normalize_text("Today is 2023-10-25.")
    assert "datetoken" in normalize_text("Date: 12/12/2022")
    
    assert "urltoken" in normalize_text("Visit https://example.com for more info")
    assert "emailtoken" in normalize_text("Contact test@example.com")
    assert "numtoken" in normalize_text("I have 100 apples")

    assert normalize_text("HELLO-World!") == "hello world"

def test_wrapper_normalize():
    df = pd.DataFrame({
        'content': ['Hello World! Visit https://example.com'],
        'title': ['My Title 123']
    })
    
    df_res = wrapper_normalize(df)
    
    assert df_res.iloc[0]['content'] == 'Hello World! Visit https://example.com'
    assert "urltoken" in df_res.iloc[0]['content_normalized']
    assert "hello world" in df_res.iloc[0]['content_normalized']
    assert "numtoken" in df_res.iloc[0]['title_normalized']
    assert "my title" in df_res.iloc[0]['title_normalized']

def test_wrapper_tokenize():
    df = pd.DataFrame({
        'content': ['this is a test sentence with running and jumping'],
        'content_normalized': ['this is a test sentence with running and jumping'],
    })
    
    df_res, local_raw, local_no_stop, local_stemmed = wrapper_tokenize(df)
    
    processed_text = df_res.iloc[0]['content_processed']
    
    assert 'test' in processed_text
    assert 'run' in processed_text
    assert 'jump' in processed_text
    assert 'this' not in processed_text
    
    assert 'running' in local_raw
    assert 'running' in local_no_stop
    assert 'run' in local_stemmed

def test_process_and_tokenize():
    from src.preprocessing import vocab_raw, vocab_no_stop, vocab_stemmed
    
    vocab_raw.clear()
    vocab_no_stop.clear()
    vocab_stemmed.clear()

    res = process_and_tokenize("running quickly")
    
    assert "run" in res
    assert "quick" in res
    
    assert "running" in vocab_raw
    assert "running" in vocab_no_stop
    assert "run" in vocab_stemmed

def test_parallel_process_dataframe():
    df = pd.DataFrame({'val': [1, 2, 3, 4]})
    res_df = parallel_process(df, dummy_func, n_cores=2)
    
    assert len(res_df) == 4
    assert list(res_df['val']) == [2, 4, 6, 8]

def test_parallel_process_tuple_return():
    df = pd.DataFrame({'val': [1, 2]})
    res_df = parallel_process(df, dummy_func_tuple, n_cores=1)
    
    assert len(res_df) == 2
    assert list(res_df['val']) == [2, 4]


def test_run_cleaning_pipeline(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    split_dir = tmp_path / "splits"

    df = pd.DataFrame({
        'Unnamed: 0': ['0', '1'],
        'id': ['100', '101'],
        'type': ['reliable', 'fake'],
        'content': [
            'This is a TEST article with https://example.com and 2023-10-25',
            'Another article about running and jumping'
        ],
        'domain': ['example.com', 'example.org'],
        'authors': ['Alice', None],
        'title': ['Title 123', None],
        'scraped_at': ['2024-01-01', '2024-01-02'],
    })
    df.to_csv(input_path, index=False)

    result_paths = run_cleaning_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        n_cores=1,
        split_output_dir=str(split_dir),
        split_prefix='cleaned',
        chunksize=1,
        print_summary=False,
    )

    assert output_path.exists()
    assert Path(result_paths['processed_path']) == output_path
    assert Path(result_paths['train_path']).exists()
    assert Path(result_paths['val_path']).exists()
    assert Path(result_paths['test_path']).exists()

    result = pd.read_csv(output_path)

    assert len(result) == 2
    assert 'content_processed' in result.columns
    assert 'content_normalized' in result.columns
    assert 'Unnamed: 0' not in result.columns
    assert 'id' in result.columns
    assert 'urltoken' in result.iloc[0]['content_normalized']
    assert 'https://example.com' in result.iloc[0]['content']
    assert 'test articl' in result.iloc[0]['content_processed']
    assert result.iloc[1]['authors'] == 'unknown_author'
    assert list(result['type']) == [0, 1]


def test_initial_cleaning_drops_invalid_scraped_at_and_reports_count(capsys):
    df = pd.DataFrame({
        'Unnamed: 0': [0, 1, 2],
        'id': ['0', '1', '2'],
        'type': ['reliable', 'fake', 'political'],
        'content': ['a', 'b', 'c'],
        'domain': ['x.com', 'y.com', 'z.com'],
        'scraped_at': ['2024-01-01', 'not-a-date', '2024-01-03'],
    })

    df_clean = initial_cleaning(df)
    captured = capsys.readouterr()

    assert len(df_clean) == 2
    assert list(df_clean['id']) == ['0', '2']
    assert "Dropped 1 rows with missing or invalid scraped_at." in captured.out


def test_initial_cleaning_drops_rows_with_missing_id():
    df = pd.DataFrame({
        'Unnamed: 0': [0, 1, 2],
        'id': ['10', None, '12'],
        'type': ['reliable', 'fake', 'political'],
        'content': ['a', 'b', 'c'],
        'domain': ['x.com', 'y.com', 'z.com'],
        'scraped_at': ['2024-01-01', '2024-01-02', '2024-01-03'],
    })

    df_clean = initial_cleaning(df)

    assert len(df_clean) == 2
    assert list(df_clean['id']) == ['10', '12']
    assert 'Unnamed: 0' not in df_clean.columns


def test_chronological_split_dataframe_orders_data_without_leakage():
    df = pd.DataFrame({
        'id': [10, 2, 8, 1, 4, 3, 7, 5, 9, 6],
        'scraped_at': [
            '2024-01-10', '2024-01-02', '2024-01-08', '2024-01-01', '2024-01-04',
            '2024-01-03', '2024-01-07', '2024-01-05', '2024-01-09', '2024-01-06'
        ],
    })

    train_df, val_df, test_df = chronological_split_dataframe(df)

    assert len(train_df) == 8
    assert len(val_df) == 1
    assert len(test_df) == 1

    assert train_df['scraped_at'].is_monotonic_increasing
    assert val_df['scraped_at'].is_monotonic_increasing
    assert test_df['scraped_at'].is_monotonic_increasing

    assert train_df['scraped_at'].max() < val_df['scraped_at'].min()
    assert val_df['scraped_at'].max() < test_df['scraped_at'].min()


def test_chronological_split_dataframe_uses_id_as_tie_breaker():
    df = pd.DataFrame({
        'id': [3, 1, 2, 5, 4, 6, 7, 8, 9, 10],
        'scraped_at': [
            '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03',
            '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08'
        ],
    })

    train_df, val_df, test_df = chronological_split_dataframe(df)

    assert list(train_df.iloc[:3]['id']) == [1, 2, 3]
    assert len(train_df) == 8
    assert len(val_df) == 1
    assert len(test_df) == 1


def test_chronological_split_dataframe_rejects_invalid_dates():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'scraped_at': ['2024-01-01', 'not-a-date', '2024-01-03'],
    })

    with pytest.raises(ValueError, match="invalid or missing datetimes"):
        chronological_split_dataframe(df)
