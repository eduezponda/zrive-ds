import unittest
from src.module_1.module_1_meteo_api import make_api_call_with_cool_off
import pytest
from unittest.mock import Mock, patch
import requests


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
        "HTTP error: Error. Esperando 1 segundos",
        "HTTP error: Error. Esperando 2 segundos",
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
            "Connection error: Error. Esperando 1 segundos",
            "Connection error: Error. Esperando 2 segundos",
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
            "Request error: Error. Esperando 1 segundos",
            "Request error: Error. Esperando 2 segundos",
        ]
        assert [r.msg for r in caplog.records] == log_messages


if __name__ == "__main__":
    unittest.main()
