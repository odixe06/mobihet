"""Successful private create may precede read visibility; retry reads, never create."""
from unittest.mock import Mock, patch

import pytest
from requests import Response
from requests.exceptions import HTTPError

from scripts.stage_r20_input import wait_ready


def http_error(status):
    response = Response(); response.status_code = status
    return HTTPError(response=response)


def test_read_visibility_lag_does_not_recreate():
    api = Mock(); api.dataset_status.side_effect = [http_error(403), "processing", "ready"]
    with patch("scripts.stage_r20_input.time.sleep") as sleep: wait_ready(api, "owner/input")
    assert api.dataset_status.call_count == 3 and sleep.call_count == 2
    api.dataset_create_new.assert_not_called()


def test_unknown_read_state_fails_bounded_without_create():
    api = Mock(); api.dataset_status.side_effect = http_error(403)
    with patch("scripts.stage_r20_input.time.sleep") as sleep, pytest.raises(RuntimeError): wait_ready(api, "owner/input")
    assert api.dataset_status.call_count == 20 and sleep.call_count == 19
    api.dataset_create_new.assert_not_called()


def test_unexpected_read_error_stops_immediately():
    api = Mock(); api.dataset_status.side_effect = http_error(401)
    with patch("scripts.stage_r20_input.time.sleep") as sleep, pytest.raises(HTTPError): wait_ready(api, "owner/input")
    sleep.assert_not_called(); api.dataset_create_new.assert_not_called()
