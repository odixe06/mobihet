"""Execute a Python helper with finite default requests timeouts; no credential output."""
import runpy
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
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
