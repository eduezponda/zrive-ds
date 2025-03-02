import pandas as pd
from module_4.utils import (
    get_feature_cols,
    push_relevant_orders,
    build_feature_frame,
    INFO_COLS,
    CATEGORICAL_COLS,
    LABEL_COL
)


def test_get_feature_cols():
    data = {
        INFO_COLS[0]: [1, 2, 3],
        CATEGORICAL_COLS[0]: [10, 20, 30],
        LABEL_COL: [0, 1, 0],
        'test1': [1, 0, 1],
        'test2': [1, 0, 1],
    }
    df = pd.DataFrame(data)

    feature_cols = get_feature_cols(df)
    
    expected_columns = ['test1', 'test2']
    
    assert feature_cols == expected_columns

def test_push_relevant_orders():
    data = {
        "order_id": [1, 1, 2, 2, 2, 2, 2],
        LABEL_COL: [0, 0, 1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    df_selected = push_relevant_orders(df)

    expected_data = {
        "order_id": [2, 2, 2, 2, 2],
        LABEL_COL: [1, 1, 1, 1, 1],
    }
    expected_df = pd.DataFrame(expected_data)
    
    pd.testing.assert_frame_equal(df_selected.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_build_feature_frame(monkeypatch):
    data = {
        "order_id": [1, 2, 3],
        LABEL_COL: [0, 1, 0],
        "created_at": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    }
    df = pd.DataFrame(data)
    
    monkeypatch.setattr('module_4.utils.load_dataset', lambda: df)
    
    df = build_feature_frame()

    expected_columns = ["created_at", "order_date"]
    assert all(col in df.columns for col in expected_columns)
    
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert pd.api.types.is_object_dtype(df["order_date"])