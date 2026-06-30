from __future__ import annotations

from pathlib import Path

import abm_webapp.app as webapp_app


def test_format_sse_event_shape() -> None:
    frame = webapp_app._format_sse_event(event="metrics_delta", payload={"run_id": "r1", "rows": []}, event_id=7)
    assert frame.startswith("id: 7\nevent: metrics_delta\n")
    assert "data: {\"run_id\":\"r1\",\"rows\":[]}" in frame
    assert frame.endswith("\n\n")


def test_read_text_delta_respects_offset_and_chunk_size(tmp_path: Path) -> None:
    p = tmp_path / "log.txt"
    p.write_text("abcdef", encoding="utf-8")

    chunk1, off1 = webapp_app._read_text_delta(p, offset=0, max_bytes=3)
    assert chunk1 == "abc"
    assert off1 == 3

    chunk2, off2 = webapp_app._read_text_delta(p, offset=off1, max_bytes=10)
    assert chunk2 == "def"
    assert off2 == 6


def test_status_signature_is_stable_for_same_payload() -> None:
    payload = {
        "run_id": "x",
        "state": "running",
        "t_last": 12,
        "message": "ok",
        "updated_at": "2026-01-01T00:00:00",
        "log_path": "a.txt",
    }
    sig1 = webapp_app._status_signature(payload)
    sig2 = webapp_app._status_signature(dict(payload))
    assert sig1 == sig2


def test_stream_endpoint_emits_snapshot_frame() -> None:
    app = webapp_app._build_dash_app()
    client = app.server.test_client()

    resp = client.get("/stream/run/unit_test_stream", buffered=False)
    try:
        first_frame = next(resp.response).decode("utf-8", errors="replace")
    finally:
        resp.close()

    assert "event: snapshot" in first_frame
    assert '"run_id":"unit_test_stream"' in first_frame


def test_live_callback_uses_inputs_only_no_state() -> None:
    app = webapp_app._build_dash_app()
    live_spec = None
    for output_id, spec in app.callback_map.items():
        if "price-graph.figure" in output_id:
            live_spec = spec
            break
    assert live_spec is not None
    assert live_spec["state"] == []
    input_keys = [(item["id"], item["property"]) for item in live_spec["inputs"]]
    assert ("stream-event-seq", "value") in input_keys
    assert ("stream-run-id", "value") in input_keys



def test_tier_render_tracking_detects_new_data() -> None:
    key = "unit-test-run"
    webapp_app._clear_tier_tracking(key)
    # First call with version=1 should render (new data, no previous render)
    assert webapp_app._should_render_tier(key, "medium", 1, min_interval_s=0.0) is True
    # Same version again should NOT render (already rendered)
    assert webapp_app._should_render_tier(key, "medium", 1, min_interval_s=0.0) is False
    # New version should render
    assert webapp_app._should_render_tier(key, "medium", 2, min_interval_s=0.0) is True
    # force=True should always render even with same version
    assert webapp_app._should_render_tier(key, "medium", 2, min_interval_s=0.0, force=True) is True
    webapp_app._clear_tier_tracking(key)
