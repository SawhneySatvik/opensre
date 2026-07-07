"""Unit tests for K8s trigger alert Datadog verification helpers."""

from __future__ import annotations

from unittest.mock import patch

from tests.e2e.kubernetes.trigger_alert import (
    DD_SINCE_EPOCH_BUFFER_SECONDS,
    TRANSFORM_ERROR_JOB,
    _build_datadog_search_payload,
    datadog_log_search_window,
    verify,
    wait_for_transform_failure,
)


def test_datadog_log_search_window_anchors_on_since_epoch() -> None:
    since_epoch = 1_700_000_000.0
    from_value, to_value = datadog_log_search_window(since_epoch)
    expected_from = int((since_epoch - DD_SINCE_EPOCH_BUFFER_SECONDS) * 1000)
    assert from_value == str(expected_from)
    assert to_value == "now"


def test_datadog_log_search_window_without_since_epoch_uses_broader_relative_window() -> None:
    from_value, to_value = datadog_log_search_window(None)
    assert from_value == "now-15m"
    assert to_value == "now"


def test_datadog_log_search_window_without_since_epoch_honors_now_epoch() -> None:
    now_epoch = 1_700_000_000.0
    from_value, to_value = datadog_log_search_window(None, now_epoch=now_epoch)
    expected_from = int((now_epoch - 15 * 60) * 1000)
    assert from_value == str(expected_from)
    assert to_value == "now"


def test_build_datadog_search_payload_includes_pipeline_error_query() -> None:
    payload = _build_datadog_search_payload(1_700_000_000.0)
    assert payload["filter"]["query"] == "kube_namespace:tracer-test PIPELINE_ERROR"  # type: ignore[index]


def test_wait_for_transform_failure_returns_true_on_failed_job() -> None:
    with patch(
        "tests.e2e.kubernetes.trigger_alert.wait_for_job",
        return_value="failed",
    ) as wait_for_job:
        assert wait_for_transform_failure(timeout=30) is True
    wait_for_job.assert_called_once_with("tracer-test", TRANSFORM_ERROR_JOB, timeout=30)


def test_wait_for_transform_failure_returns_false_on_timeout() -> None:
    with patch(
        "tests.e2e.kubernetes.trigger_alert.wait_for_job",
        side_effect=TimeoutError("timed out"),
    ):
        assert wait_for_transform_failure(timeout=30) is False


def test_verify_waits_for_transform_job_before_datadog_poll() -> None:
    with (
        patch("tests.e2e.kubernetes.trigger_alert.update_kubeconfig") as update_kubeconfig,
        patch(
            "tests.e2e.kubernetes.trigger_alert.wait_for_transform_failure",
            return_value=True,
        ) as wait_for_transform,
        patch("tests.e2e.kubernetes.trigger_alert._poll_datadog_logs", return_value=True) as poll_dd,
        patch("tests.e2e.kubernetes.trigger_alert.query_slack_alerts", return_value=True),
        patch("tests.e2e.kubernetes.trigger_alert.get_channel_id", return_value="C123"),
        patch("tests.e2e.kubernetes.trigger_alert.time.sleep") as sleep,
    ):
        assert verify(1_700_000_000.0, wait_for_transform_job=True, dd_flush_wait=30) == 0

    update_kubeconfig.assert_called_once()
    wait_for_transform.assert_called_once()
    sleep.assert_called_once_with(30)
    poll_dd.assert_called_once()
