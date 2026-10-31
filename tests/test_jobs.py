import threading
import time

from rehearsal.pipeline import JobManager


def test_lifecycle_done_and_error():
    m = JobManager()
    go = threading.Event()

    def ok(job):
        job.log("line 1\nline 2")
        job.progress("work", 0.5)
        go.wait(5)
        return {"answer": 42}

    jid = m.submit("perceive", ok, {"x": 1})
    time.sleep(0.1)
    v = m.get(jid)
    assert v["state"] == "running" and v["progress"] == {"phase": "work", "fraction": 0.5}
    assert v["log"] == ["line 1", "line 2"] and v["params"] == {"x": 1}
    go.set()
    v = m.wait(jid, 5)
    assert v["state"] == "done" and v["result"] == {"answer": 42} and v["error"] is None
    assert v["finished"] >= v["started"] >= v["created"]

    def bad(job):
        raise ValueError("boom")

    v = m.wait(m.submit("perceive", bad), 5)
    assert v["state"] == "error" and v["error"] == "ValueError: boom" and v["result"] is None
    assert any("Traceback" in line for line in v["log"])
    assert m.get("nope") is None


def test_rehearse_runs_one_at_a_time_perceive_two():
    m = JobManager()
    lock = threading.Lock()
    active = {"rehearse": 0, "perceive": 0}
    peak = {"rehearse": 0, "perceive": 0}

    def make(kind):
        def work(job):
            with lock:
                active[kind] += 1
                peak[kind] = max(peak[kind], active[kind])
            time.sleep(0.2)
            with lock:
                active[kind] -= 1
            return kind
        return work

    r = [m.submit("rehearse", make("rehearse")) for _ in range(3)]
    p = [m.submit("perceive", make("perceive")) for _ in range(4)]
    time.sleep(0.05)
    states = [m.get(j)["state"] for j in r]
    assert states == ["running", "queued", "queued"]
    assert m.get(r[2])["queue_ahead"] == 2
    for j in r + p:
        assert m.wait(j, 10)["state"] == "done"
    assert peak == {"rehearse": 1, "perceive": 2}
    assert m.latest("rehearse")["id"] == r[-1]
    assert len(m.list()) == 7 and "log" not in m.list()[0]
