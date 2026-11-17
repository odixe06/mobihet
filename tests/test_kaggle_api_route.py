"""Keep routing opt-in, HTTPS-only and limited to Kaggle's production API."""
import pytest
import requests
import runpy
import sys
from pathlib import Path
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


def test_saved_account_bootstraps_before_auth(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.setenv("MOBIHET_KAGGLE_API_VIA_WWW", "1")
    monkeypatch.setattr(KaggleHttpClient, "_get_request_url", KaggleHttpClient._get_request_url)
    monkeypatch.setattr(requests.Session, "__init__", requests.Session.__init__)
    sent = []
    monkeypatch.setattr(requests.Session, "send", lambda self, request, **kw: sent.append(kw))
    monkeypatch.setattr(requests.Session, "request", requests.Session.request)
    monkeypatch.setattr(sys, "argv", ["kaggle_as_bounded.py", "owner", "--", "python", "child.py"])
    original = runpy.run_path
    seen = []

    def run(path, **kw):
        if Path(path).name != "kaggle_as.py":
            return original(path, **kw)
        seen.append(sys.argv[1:])
        client = KaggleHttpClient(env=KaggleEnv.PROD)
        url = client._get_request_url("users.AccountService", "GenerateAccessToken")
        assert url.startswith("https://www.kaggle.com/api/v1/")
        with requests.Session() as session:
            session.send(requests.Request("POST", url).prepare())
        return {}

    monkeypatch.setattr(runpy, "run_path", run)
    original(str(Path(__file__).resolve().parents[1] / "scripts/kaggle_as_bounded.py"), run_name="__main__")
    assert seen == [["owner", "--", "python", "child.py"]]
    assert sent == [{"timeout": (10, 60)}]
