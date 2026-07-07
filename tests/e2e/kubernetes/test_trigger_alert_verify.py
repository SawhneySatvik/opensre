"""Unit tests for K8s trigger alert Datadog verification helpers."""

from __future__ import annotations

from unittest.mock import patch

from tests.e2e.kubernetes.trigger_alert import (
    DD_SINCE_EPOCH_BUFFER_SECONDS,
    EXTRACT_JOB,
    PIPELINE_TIMED_FALLBACK_SECONDS,
    TRANSFORM_ERROR_JOB,
    _build_datadog_search_payload,
    datadog_log_search_window,
    verify,
    wait_for_pipeline_inject_error,
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
        "tests.e2e.kubernetes.trigger_alert._wait_for_job_status",
        return_value="failed",
    ) as wait_for_status:
        assert wait_for_transform_failure(timeout=30) is True
    wait_for_status.assert_called_once_with(TRANSFORM_ERROR_JOB, expected="failed", timeout=30)


def test_wait_for_transform_failure_returns_false_on_timeout() -> None:
    with patch(
        "tests.e2e.kubernetes.trigger_alert._wait_for_job_status",
        return_value=None,
    ):
        assert wait_for_transform_failure(timeout=30) is False


def test_wait_for_pipeline_inject_error_submits_transform_when_missing() -> None:
    with (
        patch(
            "tests.e2e.kubernetes.trigger_alert.ensure_ci_cluster_access",
            return_value=True,
        ),
        patch("tests.e2e.kubernetes.trigger_alert.ensure_nodegroup_capacity") as ensure_nodes,
        patch(
            "tests.e2e.kubernetes.trigger_alert._wait_for_job_status",
            side_effect=["complete", "failed"],
        ) as wait_for_status,
        patch(
            "tests.e2e.kubernetes.trigger_alert._submit_transform_error_if_missing"
        ) as submit_transform,
        patch("tests.e2e.kubernetes.trigger_alert._diagnose_pipeline_jobs") as diagnose,
    ):
        assert wait_for_pipeline_inject_error() is True

    ensure_nodes.assert_called_once()
    assert wait_for_status.call_count == 2
    wait_for_status.assert_any_call(EXTRACT_JOB, expected="complete", timeout=180)
    submit_transform.assert_called_once()
    diagnose.assert_not_called()


def test_wait_for_pipeline_inject_error_returns_none_without_cluster_access() -> None:
    with patch(
        "tests.e2e.kubernetes.trigger_alert.ensure_ci_cluster_access",
        return_value=False,
    ):
        assert wait_for_pipeline_inject_error() is None


def test_wait_for_pipeline_inject_error_diagnoses_on_extract_timeout() -> None:
    with (
        patch(
            "tests.e2e.kubernetes.trigger_alert.ensure_ci_cluster_access",
            return_value=True,
        ),
        patch("tests.e2e.kubernetes.trigger_alert.ensure_nodegroup_capacity"),
        patch(
            "tests.e2e.kubernetes.trigger_alert._wait_for_job_status",
            return_value=None,
        ),
        patch("tests.e2e.kubernetes.trigger_alert._diagnose_pipeline_jobs") as diagnose,
    ):
        assert wait_for_pipeline_inject_error() is False

    diagnose.assert_called_once()


def test_verify_waits_for_transform_job_before_datadog_poll() -> None:
    with (
        patch(
            "tests.e2e.kubernetes.trigger_alert.wait_for_pipeline_inject_error",
            return_value=True,
        ) as wait_for_pipeline,
        patch(
            "tests.e2e.kubernetes.trigger_alert._poll_datadog_logs", return_value=True
        ) as poll_dd,
        patch("tests.e2e.kubernetes.trigger_alert.query_slack_alerts", return_value=True),
        patch("tests.e2e.kubernetes.trigger_alert.get_channel_id", return_value="C123"),
        patch("tests.e2e.kubernetes.trigger_alert.time.sleep") as sleep,
    ):
        assert verify(1_700_000_000.0, wait_for_transform_job=True, dd_flush_wait=30) == 0

    wait_for_pipeline.assert_called_once()
    sleep.assert_called_once_with(30)
    poll_dd.assert_called_once()


def test_verify_uses_timed_fallback_when_job_polling_unavailable() -> None:
    with (
        patch(
            "tests.e2e.kubernetes.trigger_alert.wait_for_pipeline_inject_error",
            return_value=None,
        ),
        patch("tests.e2e.kubernetes.trigger_alert._poll_datadog_logs", return_value=True),
        patch("tests.e2e.kubernetes.trigger_alert.query_slack_alerts", return_value=True),
        patch("tests.e2e.kubernetes.trigger_alert.get_channel_id", return_value="C123"),
        patch("tests.e2e.kubernetes.trigger_alert.time.sleep") as sleep,
    ):
        assert verify(1_700_000_000.0, wait_for_transform_job=True) == 0

    sleep.assert_called_once_with(PIPELINE_TIMED_FALLBACK_SECONDS)
