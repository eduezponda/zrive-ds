from module_4.train import (
    feature_label_split,
)
from module_4.utils import LABEL_COL
import pandas as pd


def test_feature_label_split():
    data = {
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [10, 20, 30, 40, 50],
        LABEL_COL: [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    X, y = feature_label_split(df)

    expected_X = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [10, 20, 30, 40, 50]
    })

    pd.testing.assert_frame_equal(X, expected_X)

    expected_y = pd.Series([0, 1, 0, 1, 0], name=LABEL_COL)
    pd.testing.assert_series_equal(y, expected_y)

