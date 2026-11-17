"""Process-scoped Kaggle API routing for an unreachable api.kaggle.com path."""
import os


def configure():
    if os.environ.get("MOBIHET_KAGGLE_API_VIA_WWW") != "1":
        return
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    import requests
    from urllib3.util.retry import Retry

    original = KaggleHttpClient._get_request_url
    if getattr(original, "_mobihet_www_route", False):
        return

    def request_url(self, service, method):
        url = original(self, service, method)
        prefix = "https://api.kaggle.com/v1/"
        return "https://www.kaggle.com/api/v1/" + url[len(prefix):] if url.startswith(prefix) else url

    request_url._mobihet_www_route = True
    KaggleHttpClient._get_request_url = request_url

    session_init = requests.Session.__init__

    def init(self):
        session_init(self)
        # Connect failures precede request delivery. Never retry a sent push after a read error.
        retry = Retry(total=None, connect=3, read=0, redirect=0, status=0, other=0, backoff_factor=.5)
        self.mount("https://www.kaggle.com/", requests.adapters.HTTPAdapter(max_retries=retry))

    requests.Session.__init__ = init
