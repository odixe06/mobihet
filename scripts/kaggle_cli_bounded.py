"""Kaggle CLI entry point with finite network timeouts; use through kaggle_as.py."""
import sys

import requests

_request, _send = requests.Session.request, requests.Session.send


def request(self, method, url, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (10, 60)
    return _request(self, method, url, **kwargs)


def send(self, req, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (10, 60)
    return _send(self, req, **kwargs)


requests.Session.request, requests.Session.send = request, send
from kaggle.cli import main  # noqa: E402

sys.argv[0] = "kaggle"
main()
