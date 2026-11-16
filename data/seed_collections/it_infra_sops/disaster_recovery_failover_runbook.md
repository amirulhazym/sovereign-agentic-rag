# Disaster Recovery & Automated Failover Runbook

## 1.1 Tiered RPO and RTO Service Level Matrix

| Service Tier | Workload Examples | Replication Strategy | RPO Target | RTO Target |
| :--- | :--- | :--- | :--- | :--- |
| Tier-1 Critical | Core Auth, Primary SQLite/Postgres Ledger, Qdrant Production | Synchronous NVMe replication across AZ-A & AZ-B | <= 5 minutes | <= 15 minutes |
| Tier-2 Standard | Internal BI Reporting, Document Archive | Asynchronous nightly encrypted snapshots | 24 hours | 4 hours |
| Tier-3 Cold Archive | Legacy Audit Logs, Compliance Backups | Immutable object storage tape/WORM sync | 7 days | 24 hours |

<!-- page: 2 -->
## 2.1 Automated Failover & Split-Brain Prevention Protocol

1. **Quorum Fencing:** Primary cluster nodes maintain distributed locks via a 3-node `etcd` quorum across Availability Zones A, B, and Witness C. If AZ-A loses heartbeat for > 30 seconds, STONITH fencing isolates the stale primary before promoting AZ-B.
2. **DNS & Load Balancer Cutover:** Internal service discovery updates `qdrant.internal` and `api.internal` records with a 15-second TTL.
3. **Post-Failover Integrity Check:** On-call engineers must run `curl -fsS http://localhost:8000/ready` and verify Qdrant collection point counts match SQLite `documents.chunk_count` totals before clearing the PagerDuty incident.
