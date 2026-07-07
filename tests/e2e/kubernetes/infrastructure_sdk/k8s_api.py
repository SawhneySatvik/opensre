"""EKS Kubernetes API helpers using IAM bearer tokens (no kubeconfig)."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from functools import lru_cache

import boto3
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest

from tests.e2e.kubernetes.infrastructure_sdk.eks import CLUSTER_NAME, REGION
from tests.shared.infrastructure_sdk.deployer import get_boto3_client

SERVICE_ACCOUNT = "etl-pipeline-sa"


@lru_cache(maxsize=1)
def _cluster_api_config() -> tuple[str, str]:
    eks = get_boto3_client("eks", REGION)
    resp = eks.describe_cluster(name=CLUSTER_NAME)
    cluster = resp["cluster"]
    return cluster["endpoint"], cluster["certificateAuthority"]["data"]


def _get_eks_token() -> str:
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are required for EKS Kubernetes API access")
    frozen = credentials.get_frozen_credentials()

    request = AWSRequest(
        method="GET",
        url="https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15",
        headers={"x-k8s-aws-id": CLUSTER_NAME},
    )
    SigV4QueryAuth(frozen, "sts", REGION, expires=60).add_auth(request)

    return "k8s-aws-v1." + base64.urlsafe_b64encode(request.url.encode()).rstrip(b"=").decode()


def k8s_request(method: str, path: str, body: dict | None = None) -> dict:
    endpoint, ca_data = _cluster_api_config()
    token = _get_eks_token()
    ca_bytes = base64.b64decode(ca_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
        f.write(ca_bytes)
        ca_file = f.name

    try:
        ctx = ssl.create_default_context(cafile=ca_file)
        url = f"{endpoint}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"K8s API error {exc.code}: {exc.read().decode()}") from exc
    finally:
        os.unlink(ca_file)


def job_exists(namespace: str, job_name: str) -> bool:
    try:
        k8s_request("GET", f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}")
        return True
    except RuntimeError as err:
        if " 404" in str(err) or '"code":404' in str(err):
            return False
        raise


def job_env_map(namespace: str, job_name: str) -> dict[str, str]:
    try:
        payload = k8s_request("GET", f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}")
    except RuntimeError:
        return {}
    containers = payload.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if not containers:
        return {}
    env: dict[str, str] = {}
    for item in containers[0].get("env", []):
        name = item.get("name")
        value = item.get("value")
        if name and value is not None:
            env[name] = str(value)
    return env


def _job_finish_status(namespace: str, job_name: str) -> str | None:
    try:
        payload = k8s_request("GET", f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}")
    except RuntimeError as err:
        if " 404" in str(err) or '"code":404' in str(err):
            return None
        raise
    for cond in payload.get("status", {}).get("conditions", []):
        if cond.get("type") == "Complete" and cond.get("status") == "True":
            return "complete"
        if cond.get("type") == "Failed" and cond.get("status") == "True":
            return "failed"
    return "pending"


def wait_for_job(namespace: str, job_name: str, timeout: int = 120) -> str:
    """Wait for a job to finish. Returns ``complete`` or ``failed``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _job_finish_status(namespace, job_name)
        if status in {"complete", "failed"}:
            return status
        time.sleep(2)
    raise TimeoutError(f"Job '{job_name}' did not finish within {timeout}s")


def delete_job(namespace: str, job_name: str) -> None:
    with contextlib.suppress(RuntimeError):
        k8s_request(
            "DELETE",
            f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}",
            body={"propagationPolicy": "Background"},
        )


def create_pipeline_job(
    namespace: str,
    *,
    job_name: str,
    stage: str,
    image_uri: str,
    env_vars: dict[str, str],
) -> None:
    env = [{"name": key, "value": value} for key, value in env_vars.items()]
    env.append({"name": "PIPELINE_STAGE", "value": stage})
    creds = boto3.Session().get_credentials()
    if creds is None:
        raise RuntimeError("AWS credentials are required to submit pipeline jobs")
    frozen = creds.get_frozen_credentials()
    env.extend(
        [
            {"name": "AWS_ACCESS_KEY_ID", "value": frozen.access_key},
            {"name": "AWS_SECRET_ACCESS_KEY", "value": frozen.secret_key},
            {"name": "AWS_SESSION_TOKEN", "value": frozen.token or ""},
            {"name": "AWS_REGION", "value": REGION},
            {"name": "AWS_DEFAULT_REGION", "value": REGION},
        ]
    )

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "etl-pipeline", "stage": stage, "tracer": "test-case"},
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {"app": "etl-pipeline", "stage": stage}},
                "spec": {
                    "serviceAccountName": SERVICE_ACCOUNT,
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": stage,
                            "image": image_uri,
                            "imagePullPolicy": "Always",
                            "env": env,
                        }
                    ],
                },
            },
        },
    }
    k8s_request("POST", f"/apis/batch/v1/namespaces/{namespace}/jobs", manifest)


def diagnose_pipeline_jobs(namespace: str, *, extract_job: str, transform_job: str) -> None:
    print("\n--- Pipeline job diagnostics (K8s API) ---")
    for path, label in (
        (f"/apis/batch/v1/namespaces/{namespace}/jobs", "jobs"),
        (f"/api/v1/namespaces/{namespace}/pods", "pods"),
    ):
        try:
            payload = k8s_request("GET", path)
            items = payload.get("items", [])
            print(f"{label}: {len(items)} resource(s)")
            for item in items:
                meta = item.get("metadata", {})
                status = item.get("status", {})
                print(
                    f"  - {meta.get('name')}: "
                    f"phase={status.get('phase', status.get('conditions', 'n/a'))}"
                )
        except RuntimeError as exc:
            print(f"{label}: error: {exc}")

    for job_name in (extract_job, transform_job):
        status = _job_finish_status(namespace, job_name)
        exists = job_exists(namespace, job_name)
        print(f"job {job_name}: exists={exists} status={status}")
