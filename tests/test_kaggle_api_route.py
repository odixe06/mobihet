"""Keep routing opt-in, HTTPS-only and limited to Kaggle's production API."""
import pytest
import requests
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.kaggle_http_client import KaggleHttpClient

from scripts.kaggle_api_route import configure


@pytest.mark.parametrize("enabled", [False, True])
def test_api_route_preserves_scope_and_path(monkeypatch, enabled):
    monkeypatch.setenv("MOBIHET_KAGGLE_API_VIA_WWW", "1" if enabled else "0")
    original = KaggleHttpClient._get_request_url
    monkeypatch.setattr(KaggleHttpClient, "_get_request_url", original)
    monkeypatch.setattr(requests.Session, "__init__", requests.Session.__init__)
    configure()
    client = KaggleHttpClient(env=KaggleEnv.PROD)
    origin = "https://www.kaggle.com/api" if enabled else "https://api.kaggle.com"
    assert client._get_request_url("kernels.KernelsApiService", "GetKernelSessionStatus") == (
        origin + "/v1/kernels.KernelsApiService/GetKernelSessionStatus")
    configure()
    assert client._get_request_url("security.OAuthClient", "IntrospectToken") == (
        origin + "/v1/security.OAuthClient/IntrospectToken")
    client._endpoint = "https://example.invalid"
    assert client._get_request_url("service", "request") == "https://example.invalid/v1/service/request"
    assert client.get_non_api_endpoint() == "https://www.kaggle.com"
    with requests.Session() as session:
        retry = session.get_adapter("https://www.kaggle.com/api/v1/service/request").max_retries
        if enabled:
            assert retry.connect == 3 and retry.read == retry.status == retry.other == 0
        else:
            assert retry.total == 0
        assert session.get_adapter("https://example.invalid").max_retries.total == 0
