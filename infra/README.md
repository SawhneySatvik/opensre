# `infra/`

Infrastructure code for opensre local development and deployment.

## What's here

| Path | Purpose |
| --- | --- |
| [`deployment/`](deployment/) | Deployment operations and external runtime entrypoints. |
| [`scripts/`](scripts/) | One-time bootstrap scripts (e.g. [`bootstrap-bench-state.sh`](scripts/bootstrap-bench-state.sh) for the Cloud-OpsBench Terraform state backend). |
| `docker-compose.*.yml` | Local development environments (database, RabbitMQ, testing). |
| `install-proxy/` | Install proxy utility. |

## Cloud-OpsBench AWS infrastructure

The Terraform module for running Cloud-OpsBench on AWS Fargate lives with the
benchmark code at
[`tests/benchmarks/cloudopsbench/infra/`](../tests/benchmarks/cloudopsbench/infra/).
See that directory's [README](../tests/benchmarks/cloudopsbench/infra/README.md)
and the benchmark runner guide at
[`tests/benchmarks/cloudopsbench/README.md`](../tests/benchmarks/cloudopsbench/README.md).
