import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    initial_cleaning,
    normalize_text,
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
        'inserted_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'updated_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'type': ['article', np.nan, 'blog', 'news'],
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
    assert 'inserted_at' not in df_clean.columns
    assert 'updated_at' not in df_clean.columns
    assert 'all_null' not in df_clean.columns

    assert len(df_clean) == 1
    assert df_clean.iloc[0]['type'] == 'article'
    assert df_clean.iloc[0]['title'] == 'no_title'
    assert df_clean.iloc[0]['authors'] == 'Alice'
    assert isinstance(df_clean.iloc[0]['content'], str)
    assert isinstance(df_clean.iloc[0]['title'], str)
    assert pd.api.types.is_datetime64_any_dtype(df_clean['scraped_at'])

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
    
    assert "urltoken" in df_res.iloc[0]['content']
    assert "hello world" in df_res.iloc[0]['content']
    assert "numtoken" in df_res.iloc[0]['title']
    assert "my title" in df_res.iloc[0]['title']

def test_wrapper_tokenize():
    df = pd.DataFrame({
        'content': ['this is a test sentence with running and jumping']
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

    df = pd.DataFrame({
        'Unnamed: 0': ['0', '1'],
        'type': ['article', 'article'],
        'content': [
            'This is a TEST article with https://example.com and 2023-10-25',
            'Another article about running and jumping'
        ],
        'domain': ['example.com', 'example.org'],
        'authors': ['Alice', None],
        'title': ['Title 123', None],
    })
    df.to_csv(input_path, index=False)

    result = run_cleaning_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        n_cores=1,
        print_summary=False,
    )

    assert output_path.exists()
    assert len(result) == 2
    assert 'content_processed' in result.columns
    assert 'Unnamed: 0' not in result.columns
    assert 'urltoken' in result.iloc[0]['content']
    assert 'test articl' in result.iloc[0]['content_processed']
    assert result.iloc[1]['authors'] == 'unknown_author'
