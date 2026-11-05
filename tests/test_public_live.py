"""Public demo with live uploads: hard cap, per-request ceiling, rate limit, fallback, purge. No network."""
import os
import time

import pytest

from rehearsal import meta_client as M
from rehearsal import public_live as PL


@pytest.fixture
def live_env(budget_dir, monkeypatch):
    monkeypatch.setenv("REHEARSAL_PUBLIC_DEMO", "1")
    monkeypatch.setenv("REHEARSAL_PUBLIC_LIVE", "1")
    monkeypatch.setenv("REHEARSAL_BUDGET_CAP", "1.00")
    return budget_dir


def test_enabled_needs_both_flags(monkeypatch):
    monkeypatch.setenv("REHEARSAL_PUBLIC_DEMO", "1")
    monkeypatch.delenv("REHEARSAL_PUBLIC_LIVE", raising=False)
    assert not PL.enabled()
    monkeypatch.setenv("REHEARSAL_PUBLIC_LIVE", "1")
    assert PL.enabled()
    monkeypatch.setenv("REHEARSAL_PUBLIC_DEMO", "0")
    assert not PL.enabled()


def test_samples_stay_recorded_uploads_go_live(live_env, monkeypatch):
    from rehearsal import app_api as A
    monkeypatch.setenv("MODEL_API_KEY", "x")
    monkeypatch.delenv("REHEARSAL_OFFLINE", raising=False)
    assert A.live_models()
    assert A.live_for("photo_abc123")
    assert not A.live_for("t6_02")
    monkeypatch.delenv("REHEARSAL_PUBLIC_LIVE")
    assert not A.live_for("photo_abc123")          # plain public demo: never live


def test_hard_cap_refuses_before_the_call(live_env):
    M._reserve(0.9, "t"); M._settle(0.9, 0.9, {"caller": "t"})
    assert not PL.budget_ok()                       # $0.10 left < one $0.25 request
    with pytest.raises(M.BudgetExceeded, match="cap"):
        M._reserve(0.2, "t")


def test_budget_ok_with_room(live_env):
    assert PL.budget_ok()


def test_request_ceiling(live_env):
    with M.request_ceiling(0.25) as c:
        M._reserve(0.10, "a"); M._settle(0.10, 0.08, {"caller": "a"})
        M._reserve(0.15, "b"); M._settle(0.15, 0.15, {"caller": "b"})
        assert c["used"] == pytest.approx(0.23)
        with pytest.raises(M.BudgetExceeded, match="per-request ceiling"):
            M._reserve(0.05, "c")
    M._reserve(0.05, "c"); M._settle(0.05, 0.05, {"caller": "c"})   # outside the request: fine
    assert M.status()["reserved"] == pytest.approx(0.0)


def test_shared_ceiling_covers_other_threads(live_env):
    import threading
    err = []
    with M.request_ceiling(0.10, shared=True):
        M._reserve(0.08, "a")

        def other():
            try:
                M._reserve(0.05, "b")
            except M.BudgetExceeded as e:
                err.append(e)
        t = threading.Thread(target=other); t.start(); t.join()
        M._settle(0.08, 0.08, {"caller": "a"})
    assert err


def test_rate_limit_per_hour_and_day():
    rl = PL.RateLimiter(per_hour=3, per_day=10)
    t = 1_000_000.0
    assert all(rl.check("1.2.3.4", t + i) is None for i in range(3))
    assert "per hour" in rl.check("1.2.3.4", t + 10)
    assert rl.check("5.6.7.8", t + 10) is None      # other visitors unaffected
    for h in range(1, 4):                            # 3 more hours, 3 each -> 9 + ...
        for i in range(3):
            rl.check("1.2.3.4", t + h * 3700 + i)
    assert "per day" in rl.check("1.2.3.4", t + 4 * 3700)
    assert rl.check("1.2.3.4", t + 86400 + 4 * 3700) is None


def test_concurrency_slot():
    assert PL.try_slot()
    if PL.CONCURRENCY == 1:
        assert not PL.try_slot()
    PL.release_slot()
    assert PL.try_slot(); PL.release_slot()


def test_purge_old_uploads(tmp_path):
    (tmp_path / "uploads").mkdir(); (tmp_path / "scenes" / "_models").mkdir(parents=True)
    old = [tmp_path / "uploads" / "a.jpg", tmp_path / "scenes" / "photo_a.json", tmp_path / "scenes" / "_models" / "photo_a.json"]
    new = tmp_path / "uploads" / "b.jpg"
    keep = tmp_path / "scenes" / "t6_02.json"
    for p in old + [new, keep]:
        p.write_text("x")
    past = time.time() - 4000
    for p in old + [keep]:
        os.utime(p, (past, past))
    assert PL.purge_uploads(tmp_path) == 3
    assert new.exists() and keep.exists() and not any(p.exists() for p in old)


def test_persistent_ledger(tmp_path, monkeypatch):
    monkeypatch.delenv("REHEARSAL_BUDGET_DIR", raising=False)
    assert PL.use_persistent_ledger(str(tmp_path / "nope")) is None
    d = PL.use_persistent_ledger(str(tmp_path))
    assert d == str(tmp_path / "rehearsal_budget") and os.environ["REHEARSAL_BUDGET_DIR"] == d


def test_config_and_upload_fallback_when_budget_out(live_env, monkeypatch):
    """Cap reached: config says so (uploads off, polite message) and the upload endpoint answers 503 with it."""
    from rehearsal import server as S
    monkeypatch.setenv("MODEL_API_KEY", "x")
    monkeypatch.delenv("REHEARSAL_OFFLINE", raising=False)
    monkeypatch.setattr(S, "PUBLIC_DEMO", True)
    M._reserve(0.9, "t"); M._settle(0.9, 0.9, {"caller": "t"})

    class H:
        headers = {}
        client_address = ("9.9.9.9", 1)
    h = H()
    cfg = S.Handler._app_config(h) if hasattr(S, "Handler") else None
    if cfg is None:
        pytest.skip("handler class not exposed")
    assert cfg["live_budget_out"] and not cfg["uploads"] and "used up" in cfg["budget_out_msg"]
    with pytest.raises(S.ApiError) as ei:
        S.Handler._app_post(h, "/api/app/upload")
    assert ei.value.args and "used up" in str(ei.value.args)
