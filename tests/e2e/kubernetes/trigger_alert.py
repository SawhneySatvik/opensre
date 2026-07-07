#!/usr/bin/env python3
"""
Fast alert trigger and verification for Kubernetes test case.

Default mode triggers the failure via trigger API URL from centralized JSON config.
Config file: tests/shared/infrastructure_sdk/outputs/tracer-k8s-trigger.json

Usage:
    python -m tests.e2e.kubernetes.trigger_alert
    python -m tests.e2e.kubernetes.trigger_alert --verify
    python -m tests.e2e.kubernetes.trigger_alert --regen-config
    python -m tests.e2e.kubernetes.trigger_alert --verify-only --since-epoch 1771466422
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import boto3

from tests.e2e.kubernetes.infrastructure_sdk.eks import (
    cluster_exists,
    ensure_nodegroup_capacity,
    update_kubeconfig,
)
from tests.e2e.kubernetes.infrastructure_sdk.local import get_pod_logs, wait_for_job
from tests.shared.infrastructure_sdk.trigger_config import (
    discover_runtime_outputs,
    load_trigger_config,
    regenerate_trigger_config,
)
from tests.shared.slack_polling import get_channel_id, poll_for_message

DEFAULT_DD_MAX_WAIT = 300
DEFAULT_SLACK_MAX_WAIT = 300
DEFAULT_POST_TRIGGER_WAIT = 0
POST_TRIGGER_WAIT_ON_504 = 90
DD_LOG_QUERY = "kube_namespace:tracer-test PIPELINE_ERROR"
DD_SINCE_EPOCH_BUFFER_SECONDS = 60
K8S_NAMESPACE = "tracer-test"
EXTRACT_JOB = "etl-extract"
TRANSFORM_ERROR_JOB = "etl-transform-error"
EXTRACT_JOB_WAIT_TIMEOUT = 180
TRANSFORM_JOB_WAIT_TIMEOUT = 180
DD_AGENT_FLUSH_WAIT_SECONDS = 30

_BASE_DIR = os.path.dirname(__file__)
_JOB_TRANSFORM_ERROR_MANIFEST = os.path.join(
    _BASE_DIR, "k8s_manifests", "job-transform-error.yaml"
)


def _trigger_via_api(trigger_api_url: str) -> dict:
    url = trigger_api_url.rstrip("/") + "/trigger?inject_error=true"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 504:
            # API Gateway timed out waiting, but Lambda may still be executing.
            return {
                "status": "accepted_timeout",
                "http_status": 504,
                "raw_body": body,
            }
        raise RuntimeError(f"HTTP {exc.code} from trigger API: {body}") from exc
    return payload if isinstance(payload, dict) else {}


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _render_eks_manifest(
    manifest_path: str,
    *,
    landing_bucket: str,
    processed_bucket: str,
    s3_key: str,
    pipeline_run_id: str,
    image_uri: str,
) -> str:
    """Compatibility helper used by test_eks flow."""
    with open(manifest_path) as f:
        content = f.read()

    rendered = (
        content.replace("{{LANDING_BUCKET}}", landing_bucket)
        .replace("{{PROCESSED_BUCKET}}", processed_bucket)
        .replace("{{S3_KEY}}", s3_key)
        .replace("{{PIPELINE_RUN_ID}}", pipeline_run_id)
        .replace("tracer-k8s-test:latest", image_uri)
        .replace("imagePullPolicy: Never", "imagePullPolicy: Always")
    )

    creds = boto3.Session().get_credentials()
    frozen = creds.get_frozen_credentials() if creds else None
    if frozen:
        region = os.environ.get("AWS_REGION", "us-east-1")
        credentials_env = (
            f"            - name: AWS_ACCESS_KEY_ID\n"
            f'              value: "{frozen.access_key}"\n'
            f"            - name: AWS_SECRET_ACCESS_KEY\n"
            f'              value: "{frozen.secret_key}"\n'
            f"            - name: AWS_SESSION_TOKEN\n"
            f'              value: "{frozen.token or ""}"\n'
            f"            - name: AWS_REGION\n"
            f'              value: "{region}"\n'
            f"            - name: AWS_DEFAULT_REGION\n"
            f'              value: "{region}"\n'
        )
        rendered = rendered.replace(
            f'            - name: PIPELINE_RUN_ID\n              value: "{pipeline_run_id}"\n',
            f"            - name: PIPELINE_RUN_ID\n"
            f'              value: "{pipeline_run_id}"\n' + credentials_env,
        )

    return rendered


def _apply_manifest(content: str) -> None:
    """Compatibility helper used by test_eks flow."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        _run(["kubectl", "apply", "-f", path])
    finally:
        os.unlink(path)


def _delete_job(job_name: str) -> None:
    """Compatibility helper used by test_eks flow."""
    _run(
        ["kubectl", "delete", "job", job_name, "-n", "tracer-test", "--ignore-not-found"],
        check=False,
    )


def _load_or_regen_trigger_config() -> dict:
    try:
        return load_trigger_config()
    except Exception:
        regenerate_trigger_config()
        return load_trigger_config()


# ---------------------------------------------------------------------------
# Datadog Logs API
# ---------------------------------------------------------------------------


def datadog_log_search_window(
    since_epoch: float | None,
    *,
    now_epoch: float | None = None,
) -> tuple[str, str]:
    """Build Datadog Logs API ``from``/``to`` bounds anchored on the trigger."""
    if since_epoch is not None:
        anchor = since_epoch - DD_SINCE_EPOCH_BUFFER_SECONDS
        return str(int(anchor * 1000)), "now"
    if now_epoch is not None:
        from_ms = int((now_epoch - 15 * 60) * 1000)
        return str(from_ms), "now"
    return "now-15m", "now"


def _build_datadog_search_payload(
    since_epoch: float | None,
    *,
    now_epoch: float | None = None,
) -> dict[str, object]:
    from_value, to_value = datadog_log_search_window(since_epoch, now_epoch=now_epoch)
    return {
        "filter": {
            "query": DD_LOG_QUERY,
            "from": from_value,
            "to": to_value,
        },
        "sort": "-timestamp",
        "page": {"limit": 1},
    }


def _poll_datadog_logs(
    max_wait: int = DEFAULT_DD_MAX_WAIT,
    *,
    since_epoch: float | None = None,
) -> bool:
    api_key = os.environ.get("DD_API_KEY", "")
    app_key = os.environ.get("DD_APP_KEY", "")
    site = os.environ.get("DD_SITE", "datadoghq.com")
    if not api_key or not app_key:
        return False

    search_payload = _build_datadog_search_payload(since_epoch)
    from_value = search_payload["filter"]["from"]  # type: ignore[index]
    to_value = search_payload["filter"]["to"]  # type: ignore[index]
    print("Polling Datadog Logs API...")
    print(f"  query={DD_LOG_QUERY!r} from={from_value!r} to={to_value!r}")
    if since_epoch is not None:
        print(f"  since_epoch={since_epoch:.0f}")

    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            payload = json.dumps(search_payload).encode()
            url = f"https://api.{site}/api/v2/logs/events/search"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "DD-API-KEY": api_key,
                    "DD-APPLICATION-KEY": app_key,
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
            if body.get("data"):
                elapsed = max_wait - int(deadline - time.monotonic())
                print(f"  Log found in Datadog ({elapsed}s)")
                return True
        except Exception as e:
            print(f"  Poll error: {e}")

        remaining = int(deadline - time.monotonic())
        print(f"  Not in DD yet... ({remaining}s remaining)")
        time.sleep(5)

    return False


def _job_exists(namespace: str, job_name: str) -> bool:
    result = _run(
        ["kubectl", "get", "job", job_name, "-n", namespace],
        check=False,
    )
    return result.returncode == 0


def _kubectl_job_env(namespace: str, job_name: str) -> dict[str, str]:
    result = _run(
        [
            "kubectl",
            "get",
            "job",
            job_name,
            "-n",
            namespace,
            "-o",
            "json",
        ],
        check=False,
    )
    if result.returncode != 0:
        return {}
    payload = json.loads(result.stdout)
    env_vars = payload.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if not env_vars:
        return {}
    env: dict[str, str] = {}
    for item in env_vars[0].get("env", []):
        name = item.get("name")
        value = item.get("value")
        if name and value is not None:
            env[name] = str(value)
    return env


def _diagnose_pipeline_jobs() -> None:
    print("\n--- Pipeline job diagnostics ---")
    for cmd in (
        ["kubectl", "get", "jobs", "-n", K8S_NAMESPACE, "-o", "wide"],
        ["kubectl", "get", "pods", "-n", K8S_NAMESPACE, "-o", "wide"],
        ["kubectl", "describe", "job", EXTRACT_JOB, "-n", K8S_NAMESPACE],
        ["kubectl", "describe", "job", TRANSFORM_ERROR_JOB, "-n", K8S_NAMESPACE],
    ):
        result = _run(cmd, check=False)
        print(f"$ {' '.join(cmd)}")
        output = (result.stdout + result.stderr).strip()
        print(output or "(no output)")
    for label in ("stage=extract", "stage=transform-error"):
        logs = get_pod_logs(K8S_NAMESPACE, label)
        if logs:
            print(f"\nPod logs ({label}):\n{logs}")


def _submit_transform_error_if_missing() -> None:
    if _job_exists(K8S_NAMESPACE, TRANSFORM_ERROR_JOB):
        return

    runtime = discover_runtime_outputs()
    if not runtime:
        raise RuntimeError(
            "Cannot submit transform job: missing EKS runtime outputs "
            "(landing/processed buckets or ECR image URI)"
        )

    extract_env = _kubectl_job_env(K8S_NAMESPACE, EXTRACT_JOB)
    pipeline_run_id = extract_env.get("PIPELINE_RUN_ID") or f"verify-{int(time.time())}"
    s3_key = extract_env.get("S3_KEY", "")

    print(
        f"Transform job missing after extract completed; submitting "
        f"{TRANSFORM_ERROR_JOB!r} (run_id={pipeline_run_id!r})"
    )
    _delete_job(TRANSFORM_ERROR_JOB)
    content = _render_eks_manifest(
        _JOB_TRANSFORM_ERROR_MANIFEST,
        landing_bucket=runtime["landing_bucket"],
        processed_bucket=runtime["processed_bucket"],
        s3_key=s3_key,
        pipeline_run_id=pipeline_run_id,
        image_uri=runtime["ecr_image_uri"],
    )
    _apply_manifest(content)


def _wait_for_job_status(
    job_name: str,
    *,
    expected: str,
    timeout: int,
) -> str | None:
    print(
        f"Waiting for {K8S_NAMESPACE}/{job_name} to reach {expected!r} "
        f"(timeout {timeout}s)..."
    )
    try:
        status = wait_for_job(K8S_NAMESPACE, job_name, timeout=timeout)
    except TimeoutError:
        print(f"  Timed out waiting for job {job_name!r}")
        return None
    if status == expected:
        print(f"  Job {job_name!r} reached {expected!r}")
        return status
    print(f"  Job {job_name!r} finished with unexpected status: {status}")
    return status


def wait_for_transform_failure(timeout: int = TRANSFORM_JOB_WAIT_TIMEOUT) -> bool:
    """Wait until the error-path transform job fails on EKS."""
    status = _wait_for_job_status(TRANSFORM_ERROR_JOB, expected="failed", timeout=timeout)
    return status == "failed"


def wait_for_pipeline_inject_error() -> bool:
    """Wait for extract to complete, then for transform-error to fail on EKS.

  After an API Gateway 504 the Lambda may time out before submitting transform.
  If extract finishes on-cluster but transform is missing, submit transform-error
  from the verify step (same flow as ``test_eks.py``).
    """
    print("Ensuring EKS node group is active before polling jobs...")
    ensure_nodegroup_capacity()

    extract_status = _wait_for_job_status(
        EXTRACT_JOB,
        expected="complete",
        timeout=EXTRACT_JOB_WAIT_TIMEOUT,
    )
    if extract_status != "complete":
        _diagnose_pipeline_jobs()
        return False

    try:
        _submit_transform_error_if_missing()
    except Exception as exc:
        print(f"ERROR: could not ensure transform job exists: {exc}")
        _diagnose_pipeline_jobs()
        return False

    transform_status = _wait_for_job_status(
        TRANSFORM_ERROR_JOB,
        expected="failed",
        timeout=TRANSFORM_JOB_WAIT_TIMEOUT,
    )
    if transform_status != "failed":
        _diagnose_pipeline_jobs()
        return False
    return True


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

_SLACK_KEYWORDS = ["PIPELINE_ERROR", "Pipeline error", "tracer"]


def query_slack_alerts(
    max_wait: int = DEFAULT_SLACK_MAX_WAIT,
    channel_id: str | None = None,
    since_epoch: float | None = None,
) -> bool:
    return poll_for_message(
        _SLACK_KEYWORDS,
        channel_id=channel_id,
        max_wait=max_wait,
        since_epoch=since_epoch,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def verify(
    since_epoch: float,
    *,
    dd_max_wait: int = DEFAULT_DD_MAX_WAIT,
    slack_max_wait: int = DEFAULT_SLACK_MAX_WAIT,
    post_trigger_wait: int = DEFAULT_POST_TRIGGER_WAIT,
    wait_for_transform_job: bool = False,
    dd_flush_wait: int = DD_AGENT_FLUSH_WAIT_SECONDS,
) -> int:
    """Poll Datadog and Slack to confirm the pipeline failure was observed.

    Returns 0 on success, 1 if Datadog verification fails.
    """
    start = time.monotonic()

    if wait_for_transform_job:
        update_kubeconfig()
        if not wait_for_pipeline_inject_error():
            print(
                f"\nFAIL: pipeline inject-error path did not complete "
                f"(extract {EXTRACT_JOB_WAIT_TIMEOUT}s + transform "
                f"{TRANSFORM_JOB_WAIT_TIMEOUT}s)"
            )
            return 1
        if dd_flush_wait > 0:
            print(f"Waiting {dd_flush_wait}s for Datadog Agent to flush logs...")
            time.sleep(dd_flush_wait)
    elif post_trigger_wait > 0:
        print(
            f"Waiting {post_trigger_wait}s for pipeline + Datadog indexing "
            "(post-trigger backoff)..."
        )
        time.sleep(post_trigger_wait)

    dd_found = _poll_datadog_logs(max_wait=dd_max_wait, since_epoch=since_epoch)
    dd_elapsed = time.monotonic() - start

    if not dd_found:
        from_value, to_value = datadog_log_search_window(since_epoch)
        print(f"\nFAIL: PIPELINE_ERROR not found in Datadog within {dd_max_wait}s")
        print(f"  since_epoch={since_epoch:.0f}")
        print(f"  datadog_query={DD_LOG_QUERY!r}")
        print(f"  datadog_from={from_value!r} datadog_to={to_value!r}")
        print(f"  post_trigger_wait={post_trigger_wait}s dd_max_wait={dd_max_wait}s")
        print(f"  wait_for_transform_job={wait_for_transform_job}")
        return 1

    print(f"\nLog confirmed in Datadog ({dd_elapsed:.1f}s)")
    print("Waiting for Datadog monitor to fire and post to Slack...")

    channel_id = get_channel_id()
    slack_found = query_slack_alerts(
        max_wait=slack_max_wait,
        channel_id=channel_id,
        since_epoch=since_epoch,
    )

    total = time.monotonic() - start
    if dd_found and slack_found:
        print(f"\nEnd-to-end verified: pipeline failure -> Datadog -> Slack ({total:.1f}s)")
    else:
        print(f"\nPartial: log in Datadog but Slack alert not confirmed ({total:.1f}s)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="K8s trigger alert and verifier")
    parser.add_argument(
        "--regen-config", action="store_true", help="Regenerate centralized trigger config JSON"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify logs in DD + wait for DD alert in Slack"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Skip trigger, only run DD + Slack verification"
    )
    parser.add_argument(
        "--since-epoch",
        type=float,
        default=None,
        help="Unix timestamp to anchor Datadog/Slack verification",
    )
    parser.add_argument(
        "--dd-max-wait",
        type=int,
        default=DEFAULT_DD_MAX_WAIT,
        help=f"Seconds to poll Datadog (default: {DEFAULT_DD_MAX_WAIT})",
    )
    parser.add_argument(
        "--post-trigger-wait",
        type=int,
        default=DEFAULT_POST_TRIGGER_WAIT,
        help="Seconds to wait before Datadog polling (for async Lambda after API 504)",
    )
    parser.add_argument(
        "--wait-for-transform-job",
        action="store_true",
        help=(
            "Wait for etl-transform-error to fail on EKS before polling Datadog "
            "(requires kubectl access to tracer-eks-test)"
        ),
    )
    args = parser.parse_args()

    if args.regen_config:
        try:
            path = regenerate_trigger_config()
            print(f"Trigger config regenerated: {path}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1

    since_epoch = args.since_epoch or time.time()

    if args.verify_only:
        return verify(
            since_epoch,
            dd_max_wait=args.dd_max_wait,
            post_trigger_wait=args.post_trigger_wait,
            wait_for_transform_job=args.wait_for_transform_job,
        )

    start_epoch = time.time()
    try:
        cfg = _load_or_regen_trigger_config()
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("Run: make regen-trigger-config")
        return 1

    trigger_api_url = cfg["trigger_api_url"]
    if not cluster_exists():
        print("ERROR: EKS cluster 'tracer-eks-test' is not available.")
        print("Trigger API exists but cannot run pipeline jobs without the cluster.")
        return 1
    print("Ensuring EKS node group is active before trigger...")
    ensure_nodegroup_capacity()
    print(f"Triggering pipeline via API: {trigger_api_url}")
    try:
        response = _trigger_via_api(trigger_api_url)
    except Exception as exc:
        print(f"ERROR: API trigger failed: {exc}")
        return 1

    print(f"Trigger response: {json.dumps(response)}")

    if not args.verify:
        print("Done. DD monitor will fire in ~1-2 min -> Slack alert follows.")
        return 0

    post_trigger_wait = args.post_trigger_wait
    if response.get("status") == "accepted_timeout" and post_trigger_wait <= 0:
        post_trigger_wait = POST_TRIGGER_WAIT_ON_504
        print(f"Trigger returned 504; waiting {post_trigger_wait}s before verify")

    return verify(
        start_epoch,
        dd_max_wait=args.dd_max_wait,
        post_trigger_wait=post_trigger_wait,
        wait_for_transform_job=args.wait_for_transform_job,
    )


if __name__ == "__main__":
    sys.exit(main())
