"""Client for a bytenoise agent (e.g. the GPU box over Tailscale)."""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def _store():
    return Path(os.environ.get("BYTENOISE_REMOTES") or (Path.home() / ".bytenoise" / "remotes.json"))


def load_remotes():
    f = _store()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def add_remote(name, url, token=None):
    r = load_remotes()
    r[name] = {"url": url.rstrip("/"), "token": token}
    f = _store()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(r, indent=1), encoding="utf-8")
    try:
        os.chmod(f, 0o600)  # holds a bearer token
    except OSError:
        pass


def call(name, path, body=None, timeout=30):
    r = load_remotes().get(name)
    if r is None:
        raise KeyError(f"unknown remote {name!r}; add it with `bn remote add`")
    headers = {"Content-Type": "application/json"}
    if r.get("token"):
        headers["Authorization"] = f"Bearer {r['token']}"
    req = urllib.request.Request(f"{r['url']}/api/{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers=headers, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", "")
        except ValueError:
            msg = ""
        raise RuntimeError(f"{name}: HTTP {e.code} {msg}") from None
    except OSError as e:  # URLError subclasses OSError
        raise RuntimeError(f"{name}: cannot reach {r['url']} ({getattr(e, 'reason', e)})") from None
