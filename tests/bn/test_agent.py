import http.client
import json
import threading
import time

import pytest

from bytenoise.serve import jobs
from bytenoise.serve.agent import make_server


@pytest.fixture
def server(home):
    httpd, state = make_server("127.0.0.1", 0, token="s3cret")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def call(port, method, path, body=None, headers=None, host=None, token="s3cret"):
    h = {"Host": host or f"127.0.0.1:{port}"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if body is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request(method, path, body=None if body is None else json.dumps(body), headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def test_ui_and_health(server):
    s, html = call(server, "GET", "/", token=None)
    assert s == 200 and b"ByteNoise" in html
    assert call(server, "GET", "/api/health")[1]["ok"] is True


def test_auth_required(server):
    assert call(server, "GET", "/api/runs", token=None)[0] == 401
    assert call(server, "GET", "/api/runs", token="wrong")[0] == 401


def test_host_and_origin_checks(server):
    assert call(server, "GET", "/api/health", host="evil.example")[0] == 403
    assert call(server, "GET", "/api/health", headers={"Origin": "http://evil.example"})[0] == 403
    assert call(server, "GET", "/api/health", headers={"Origin": f"http://127.0.0.1:{server}"})[0] == 200


def test_post_needs_json_content_type(server):
    c = http.client.HTTPConnection("127.0.0.1", server)
    c.request("POST", "/api/jobs", body="experiment=x", headers={"Host": f"127.0.0.1:{server}", "Authorization": "Bearer s3cret", "Content-Type": "text/plain"})
    assert c.getresponse().status == 415


def test_no_token_means_open_only_on_loopback(home):
    with pytest.raises(SystemExit):
        make_server("0.0.0.0", 0)


@pytest.mark.parametrize("path", ["/api/runs/..%2f..%2fetc", "/api/runs/..", "/api/runs/nope", "/api/evals/../x/y.json", "/api/jobs/not-an-id"])
def test_path_traversal_and_unknown_ids(server, path):
    s, body = call(server, "GET", path)
    assert s in (400, 404) and "error" in body


def test_run_listing_metrics_and_datasets(server, home):
    d = home / "runs" / "r1"
    d.mkdir()
    (d / "metrics.jsonl").write_text('{"update": 1, "loss": 2.0}\n{"update": 2, "loss": 1.0}\n')
    s, runs = call(server, "GET", "/api/runs")
    assert s == 200 and [r["id"] for r in runs["runs"]] == ["r1"]
    assert call(server, "GET", "/api/runs/r1/metrics")[1]["series"]["loss"] == [2.0, 1.0]
    assert [x["name"] for x in call(server, "GET", "/api/datasets")[1]] == ["dummy-v1"]
    assert "quick" in call(server, "GET", "/api/protocols")[1]


def test_job_end_to_end_then_generate_and_eval(server, home):
    s, job = call(server, "POST", "/api/jobs", {"experiment": "ar_debug_smoke", "device": "cpu"})
    assert s == 200 and job["status"] in ("queued", "running")
    jid = job["id"]
    for _ in range(120):
        job = call(server, "GET", f"/api/jobs/{jid}")[1]
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(1)
    assert job["status"] == "done", call(server, "GET", f"/api/jobs/{jid}/log")[1]
    run = job["runs"][0]
    assert call(server, "GET", f"/api/runs/{run}")[1]["info"]["status"] == "done"
    s, g = call(server, "POST", "/api/generate", {"run": run, "prompt": "the ", "n_bytes": 12, "seed": 1})
    assert s == 200 and len(g["repr"]) > 0 and g["model"]["family"] == "ar_lm"
    s, e = call(server, "POST", "/api/eval", {"run": run, "protocol": "quick"})
    assert s == 200 and e["result"]["results"]["bpb"]["exact"] is True
    assert call(server, "GET", f"/api/runs/{run}")[1]["evals"][0]["file"] == "quick@ema.json"


def test_submit_validates_experiment_and_device(server):
    assert call(server, "POST", "/api/jobs", {"experiment": "../../etc/passwd"})[0] == 400
    assert call(server, "POST", "/api/jobs", {"experiment": "nope"})[0] == 400
    assert call(server, "POST", "/api/jobs", {"experiment": "ar_debug_smoke", "device": "tpu"})[0] == 400
    assert call(server, "POST", "/api/jobs", {"experiment": "ar_debug_smoke", "sets": ["train.bogus=1"]})[0] == 400


def test_generate_on_diffusion_with_prompt_is_a_clean_error(server, home):
    from bytenoise.train.runner import run_experiment
    run = run_experiment("debug_smoke", ["train.max_updates=2"], device="cpu")[0]
    s, body = call(server, "POST", "/api/generate", {"run": run, "prompt": "hi", "n_bytes": 8})
    assert s == 400 and "unconditional" in body["error"]


def test_stop_queued_job(home):
    j = jobs.submit("ar_debug_smoke", [], "cpu")
    assert j["status"] == "queued"
    assert jobs.stop(j["id"])["status"] == "stopped"
    assert jobs.schedule() is None  # a stopped job is never started
    assert jobs.get(j["id"])["status"] == "stopped"
