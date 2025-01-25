import numpy as np
from src.module_1.module_1_meteo_api import (
    make_api_call_with_cool_off,
    get_processed_df_by_month
)
import pytest
from unittest.mock import Mock, patch
import requests
import pandas as pd


class MockResponse:
    def __init__(self, data, status_code):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code < 200 or self.status_code > 299:
            raise requests.exceptions.HTTPError("Error")

    def get(self):
        raise requests.exceptions.ConnectionError("Error")


def test_get_make_api_call_with_cool_off(monkeypatch):
    headers = {}
    mocked_response = Mock(return_value=MockResponse("mocked_response", 200))
    monkeypatch.setattr(requests, "get", mocked_response)

    response = make_api_call_with_cool_off("mock_url", headers)
    assert response == "mocked_response"


def test_post_make_api_call_with_cool_off(monkeypatch):
    headers = {}
    mocked_response = Mock(return_value=MockResponse("mocked_response", 200))
    monkeypatch.setattr(requests, "post", mocked_response)

    response = make_api_call_with_cool_off("mock_url", headers, payload="mock")
    assert response == "mocked_response"


def test_make_api_call_with_cool_off_404(monkeypatch, caplog):
    headers = {}
    mocked_response = Mock(return_value=MockResponse("mocked_response", 404))
    monkeypatch.setattr(requests, "get", mocked_response)

    with pytest.raises(requests.exceptions.HTTPError):
        make_api_call_with_cool_off("mock_url", headers)

    log_messages = ["Error 404 Not Found"]
    assert [r.msg for r in caplog.records] == log_messages


def test_make_api_call_with_cool_off_444(monkeypatch, caplog):
    headers = {}
    mocked_response = Mock(return_value=MockResponse("mocked_response", 444))
    monkeypatch.setattr(requests, "get", mocked_response)

    with (
        patch('time.sleep'),
        pytest.raises(requests.exceptions.HTTPError)
    ):
        make_api_call_with_cool_off("mock_url", headers)

    log_messages = [
        "HTTP error: Error. Waiting 1 seconds",
        "HTTP error: Error. Waiting 2 seconds",
    ]
    assert [r.msg for r in caplog.records] == log_messages


def test_make_api_call_with_cool_off_connection_error(caplog):
    with (
        patch(
            'requests.get',
            side_effect=requests.exceptions.ConnectionError("Error")
        ),
        patch('time.sleep'),
        pytest.raises(requests.exceptions.ConnectionError)
    ):
        headers = {}

        make_api_call_with_cool_off("mock_url", headers)

        log_messages = [
            "Connection error: Error. Waiting 1 seconds",
            "Connection error: Error. Waiting 2 seconds",
        ]
        assert [r.msg for r in caplog.records] == log_messages


def test_make_api_call_with_cool_off_request_exception(caplog):
    with (
        patch(
            'requests.get',
            side_effect=requests.exceptions.RequestException("Error")
        ),
        patch('time.sleep'),
        pytest.raises(requests.exceptions.RequestException)
    ):
        headers = {}

        make_api_call_with_cool_off("mock_url", headers)

        log_messages = [
            "Request error: Error. Waiting 1 seconds",
            "Request error: Error. Waiting 2 seconds",
        ]
        assert [r.msg for r in caplog.records] == log_messages


def test_get_processed_df_by_month():
    previous_data = {
        'time': ['2025-01-22 00:00', '2025-01-22 01:00', '2025-01-22 02:00'],
        'city': ['Barcelona', 'Madrid', 'Sevilla'],
        'temperature_2m_mean': [15, 14, 13],
        'precipitation_sum': [0, 1, 0],
        'wind_speed_10m_max': [5, 4, 6]
    }
    previous_df = pd.DataFrame(previous_data)

    expected_data = {
        'city': ['Barcelona', 'Madrid', 'Sevilla'],
        'month': [pd.Timestamp('2025-01-01')] * 3,
        'temperature_2m_mean_max': [15, 14, 13],
        'temperature_2m_mean_min': [15, 14, 13],
        'temperature_2m_mean_mean': [15.0, 14.0, 13.0],
        'temperature_2m_mean_std': [np.nan, np.nan, np.nan],
        'precipitation_sum_max': [0, 1, 0],
        'precipitation_sum_min': [0, 1, 0],
        'precipitation_sum_mean': [0.0, 1.0, 0.0],
        'precipitation_sum_std': [np.nan, np.nan, np.nan],
        'wind_speed_10m_max_max': [5, 4, 6],
        'wind_speed_10m_max_min': [5, 4, 6],
        'wind_speed_10m_max_mean': [5.0, 4.0, 6.0],
        'wind_speed_10m_max_std': [np.nan, np.nan, np.nan]
    }
    expected_df = pd.DataFrame(expected_data)

    response = get_processed_df_by_month(previous_df)

    pd.testing.assert_frame_equal(expected_df, response)
