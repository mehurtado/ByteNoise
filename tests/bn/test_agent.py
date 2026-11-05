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


# --- regressions from the adversarial review ---

def test_sets_cannot_point_training_at_arbitrary_directories(server, tmp_path):
    for sets in (["train.data=", f"train.data_dir={tmp_path}"], [f"train.data={tmp_path}"], ["trian.max_updates=5"]):
        s, _ = call(server, "POST", "/api/jobs", {"experiment": "ar_debug_smoke", "sets": sets})
        assert s == 400, sets


def test_tailnet_header_alone_does_not_bypass_token(home):
    httpd, _ = make_server("127.0.0.1", 0, token="s3cret", tailnet_users=["me@x.com"])
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        assert call(port, "GET", "/api/jobs", token=None, headers={"Tailscale-User-Login": "me@x.com"})[0] == 401
        assert call(port, "GET", "/api/jobs")[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_negative_content_length_and_odd_inputs_are_4xx(server):
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    c.request("POST", "/api/jobs", headers={"Host": f"127.0.0.1:{server}", "Authorization": "Bearer s3cret",
                                            "Content-Type": "application/json", "Content-Length": "-1"})
    assert c.getresponse().status == 400
    assert call(server, "GET", "/api/runs/x/metrics?max=0")[0] in (400, 404)
    assert call(server, "GET", "/api/health", token="töken")[0] == 401


def test_generate_rejects_cuda_and_failed_load_does_not_poison_cache(server):
    assert call(server, "POST", "/api/generate", {"run": "nope", "device": "cuda"})[0] == 400
    assert call(server, "POST", "/api/generate", {"run": "nope"})[0] == 400
    assert call(server, "POST", "/api/generate", {"run": "nope"})[0] == 400  # not a 500 from a None cache entry


def test_ui_is_not_frameable(server):
    c = http.client.HTTPConnection("127.0.0.1", server)
    c.request("GET", "/", headers={"Host": f"127.0.0.1:{server}"})
    r = c.getresponse()
    assert r.getheader("X-Frame-Options") == "DENY" and "frame-ancestors 'none'" in r.getheader("Content-Security-Policy")


def test_junk_job_file_does_not_wedge_queue(home):
    (jobs.jobs_dir() / "20250101-000000-abcd.json").write_text("")
    (jobs.jobs_dir() / "20250101-000001-abcd.json").write_text("[1,2]")
    assert jobs.list_jobs() == [] and jobs.schedule() is None


def test_schedule_lock_is_exclusive_across_holders(home):
    with jobs._schedule_lock() as a:
        assert a
        with jobs._schedule_lock(timeout=0.3) as b:
            assert not b


def test_recycled_pid_is_not_alive(home):
    import os
    me = {"pid": os.getpid(), "pid_start": jobs.proc_start(os.getpid())}
    assert jobs.job_process_alive(me)
    assert not jobs.job_process_alive({**me, "pid_start": me["pid_start"] + 12345})
