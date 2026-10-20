# Datasets

Three public IDS corpora spanning a decade of capture methodology, all obtainable **without any
registration form** (see the availability check in [PROGRESS.md](PROGRESS.md)). Downloads are
pinned by size and SHA-256 (`ids_anomaly.data.fetch`).

| | NSL-KDD | UNSW-NB15 | CIC-IDS2017 |
|---|---|---|---|
| Year / origin | 2009 (re-cut of KDD'99) | 2015, ACCS Cyber Range (IXIA PerfectStorm) | 2017, CIC testbed, 5 working days |
| Features used | 41 hand-crafted, incl. content & host-based | 34 Argus/Bro flow statistics → 50 after encoding | 69 CICFlowMeter statistics (+protocol) → 69 after dropping constants |
| Official split | 125,973 / 22,544 | 175,341 / 82,332 | none (we define a temporal one) |
| Attack families | DoS, probe, R2L, U2R | 9 (generic, exploits, fuzzers, DoS, recon, analysis, backdoor, shellcode, worms) | DoS, DDoS, port scan, brute force, web, bot, infiltration, Heartbleed |
| Timestamps | no | no | yes (drives the drift study) |
| Source used | GitHub `defcom17/NSL_KDD` | HF `wwydmanski/UNSW-NB15` | HF `rdpahalavan/CIC-IDS2017` |

## Protocols

* **NSL-KDD**: official train/test; the test split deliberately contains attack types unseen in
  training.
* **UNSW-NB15**: official partition. The mirror's file names are inverted (its `train.parquet`
  is the official 82,332-row *testing* set), so the loader identifies files by row count and
  fails loudly if the counts are not the official ones.
* **CIC-IDS2017** (`split="temporal"`, default): train on **Mon + Tue** (Monday is attack-free,
  Tuesday adds FTP/SSH-Patator at ~1.4% contamination: a realistic training window), test on
  **Wed–Fri**. Almost every test attack family (DoS, DDoS, port scan, web, bot, infiltration,
  Heartbleed) is *novel* to the training window. `split="iid"` is the random-split contrast.
  Subsampling keeps every class with ≤ 2,000 rows whole and down-samples the rest by one global
  ratio (150k train / ~250k test flows by default), so prevalence of rare attacks is slightly
  tilted upward; Heartbleed (11), Infiltration (36) and web-attack classes survive intact.

## Preprocessing

* NSL-KDD: `log1p` on skewed counts, standardize, one-hot categoricals (see methodology).
* UNSW-NB15 / CIC-IDS2017 (`ids_anomaly.data.tabular`): non-finite → 0, **signed** `log1p`
  (these datasets contain negative sentinels and `inf` rates), standardize, **clip at ±10σ**,
  drop training-constant columns; rare categorical levels fold into an "infrequent" bucket.
  UNSW's `stcpb`/`dtcpb` (TCP initial sequence numbers) are dropped as identifiers; CIC's flow
  id, IPs, ports and timestamp are never features (IP/port identity is a well-known shortcut).
* CIC-IDS2017's timestamp column mixes `dd/mm/yyyy HH:MM:SS` and `d/m/yyyy H:MM` and uses a
  12-hour clock without AM/PM; hours below 08:00 are shifted to the afternoon
  (`ids_anomaly.data.cicids.parse_timestamps`).

## Harmonized view (cross-dataset transfer)

UNSW-NB15 and CIC-IDS2017 are mapped onto 18 physically aligned quantities (duration in
seconds; per-direction packets/bytes; packet and byte rates recomputed from totals; mean packet
sizes; inter-arrival time and jitter in ms; forward fractions; TCP/UDP/other). It is lossy by
design, and NSL-KDD is excluded (its content/host features have no counterpart).
