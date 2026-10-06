---
name: submit-job
description: Make long-running jobs observable and recoverable with progress tracking, checkpoints, resource monitoring, and a final summary. Use when the user says submit, launch, nohup, train, collect, download, crawl, ingest, preprocess, evaluate, infer, sweep, or run a job, or asks for checkpoints, metrics, progress, or recovery on any long-running workload.
---

# Submit a job

## Job observability, progress, checkpoints, and recovery

Every long-running job must expose enough information to answer:

```text
Is it alive?
What is it doing?
How far has it progressed?
How fast is it progressing?
What resources is it using?
What has completed?
What has failed?
Can it resume safely?
Where are its outputs?
```

Do not assume the existing script provides adequate observability. Inspect it before launch.

If progress tracking, checkpointing, or recovery is missing and can be safely added, add it before submitting the job.

The exact metrics and checkpoint strategy depend on the task.

---

## Universal progress tracking

Every long-running job should record:

```text
start time
current stage
items/tasks completed
items/tasks remaining, when known
success count
failure count
elapsed time
processing rate
ETA, when estimable
last successful progress time
resource allocation
output location
```

Prefer both:

```text
human-readable logs
+
machine-readable progress/state
```

For example:

```text
<run-dir>/
├── job.log
├── progress.json
├── summary.json
├── job.pid
├── launch.txt
├── environment.txt
├── checkpoints/
├── logs/
├── artifacts/
└── failures/
```

Not every directory is required if it is irrelevant to the task.

---

## Progress file

For jobs where meaningful units of work exist, maintain something equivalent to:

```json
{
  "status": "running",
  "stage": "collecting",
  "started_at": "2026-10-02T17:00:00",
  "updated_at": "2026-10-02T18:15:00",
  "completed": 18420,
  "failed": 37,
  "total": 50000,
  "rate_per_second": 4.8,
  "percent_complete": 36.84,
  "eta_seconds": 6580
}
```

Update it periodically during execution, not only when the job finishes.

Write state atomically when practical:

```text
progress.tmp
→ rename
→ progress.json
```

This prevents an interrupted write from corrupting the only progress record.

---

## Data collection / downloading / crawling / API ingestion

For data collection jobs, checkpoint progress continuously.

Do not require the job to restart from item zero after an SSH disconnect, rate limit, network failure, machine restart, or API error when safe resumability can reasonably be implemented.

### Track collection progress

Where applicable, track:

```text
items discovered
items attempted
items successfully collected
items skipped
items already existing
items failed
items retried

bytes downloaded
records written
files written

requests made
successful requests
failed requests

HTTP/API status counts

current source
current page
current cursor
current offset
current batch
current shard

requests/second
items/second
MB/second

elapsed time
ETA
```

Also track domain-specific quantities when useful.

Examples:

```text
countries completed
cities completed
metros completed
URLs completed
API pages completed
database rows inserted
database rows updated
files processed
objects downloaded
tiles downloaded
samples collected
```

Do not invent meaningless counters simply to create more metrics.

### Data collection checkpoints

Persist enough state to safely resume.

Depending on the source, checkpoints may contain:

```text
completed IDs
pending IDs
failed IDs

pagination cursor
page number
offset
continuation token

last processed timestamp
last processed primary key

completed cities
completed countries
completed shards

download manifest
database commit position
```

Prefer deterministic work identifiers.

Examples:

```text
city_id
URL
object key
database primary key
tile coordinate
record ID
API cursor
```

A data collection job should normally be idempotent where practical:

```text
already completed item
→ verify it
→ skip it
```

rather than downloading or inserting it again.

### Manifest

For large collection jobs, maintain a manifest such as:

```text
manifest.csv
```

or:

```text
manifest.jsonl
```

Example:

```text
item_id,status,attempts,size_bytes,started_at,completed_at,error
US_NYC,done,1,84219233,...,...,
US_SF,done,2,67381922,...,...,
DE_BERLIN,failed,3,0,...,...,timeout
```

This should make it possible to identify exactly:

```text
what succeeded
what failed
what remains
```

without parsing the entire log.

### Retry behavior

Transient failures should normally be retried.

Where applicable use:

```text
bounded retries
exponential backoff
jitter
rate-limit handling
timeouts
```

Distinguish between:

```text
transient failures
permanent failures
invalid input
rate limiting
authentication failures
```

Do not retry permanently invalid requests forever.

Persist final failures to something like:

```text
failures.jsonl
```

so they can be retried separately later.

### Database ingestion

For jobs writing to PostgreSQL, databases, object stores, or other persistent systems:

track where applicable:

```text
rows read
rows inserted
rows updated
rows skipped
rows rejected
rows/sec

batch number
transaction count
commit count
database errors
duplicate/conflict count
```

Prefer bounded batches rather than one enormous transaction.

Make operations idempotent where practical.

For PostgreSQL, appropriate patterns may include:

```text
primary keys
unique constraints
ON CONFLICT
staging tables
transactional batches
```

Do not treat a database row count alone as proof that the collection is complete.

Maintain a collection/job manifest separately when appropriate.

---

## ML training / optimization

For iterative model training, preserve:

```text
best checkpoint
latest resumable checkpoint
training state
metrics history
```

At minimum, checkpoints should contain as much relevant state as supported:

```text
model weights
optimizer state
scheduler state
AMP/scaler state
epoch or step
best metric
early stopping state
configuration
seed
```

When useful for exact resume, also preserve RNG state.

Never overwrite the only recoverable checkpoint.

Use atomic writes where practical:

```text
write temporary checkpoint
→ flush/close
→ rename to final checkpoint filename
```

Track task-relevant metrics such as:

```text
train loss
validation loss
learning rate

accuracy
precision
recall
F1
AUROC
AUPRC

MAE
RMSE
R²

task-specific objectives
```

as applicable.

Do not blindly compute every known ML metric.

Track metrics useful for:

```text
optimization
comparison
diagnosis
final evaluation
```

### Best-model tracking

Determine the primary validation objective from the task. Infer whether the metric should be minimized or maximized from its semantics or the existing code/configuration — do not assume higher is always better.

Save the best checkpoint separately from the latest checkpoint. Record best metric name, value, epoch/step, and path.

### Structured metric files

In addition to human-readable logs, persist machine-readable history such as `metrics.csv`, `history.json`, or `metrics.jsonl`. Keep the file updated during training rather than writing it only after successful completion.

### Experiment tracking systems

If the repository already uses an experiment tracker (W&B, TensorBoard, MLflow, etc.), preserve it. Do not introduce an external service that requires credentials or network access unless the user asks for it. Local logs and structured metric files must still exist even when an external tracker is used.

### Multi-fold / multi-seed / distributed runs

For jobs containing folds, seeds, trials, workers, ranks, datasets, or independent tasks, keep their outputs separate. Each independent unit should expose its own progress and completion state.

Example:

```text
runs/<job>/
├── job.log
├── progress.json
├── summary.json
├── metrics.csv
├── checkpoints/
├── fold_00/
│   ├── train.log
│   ├── metrics.csv
│   ├── best.pt
│   ├── last.pt
│   └── DONE
├── fold_01/
│   └── ...
└── ...
```

### Resume behavior

If the training framework supports resume, make the latest checkpoint resumable. Before restarting an interrupted run, inspect last checkpoint, configuration, completed epochs/steps, optimizer/scheduler state, and DONE/FAILED markers. Resume only when the checkpoint and current configuration are compatible. If compatibility is uncertain, create a new run instead of silently resuming with mismatched settings.

---

## Evaluation / inference jobs

For large inference or evaluation runs, track where applicable:

```text
samples processed
samples remaining
samples/sec
failed samples

batches completed
per-dataset progress
per-shard progress

prediction output location
metric accumulation state
```

Persist intermediate predictions periodically when practical.

A failure near the end of a million-item inference run should not require repeating all million items if the workload can safely resume.

---

## Preprocessing / ETL / feature extraction

Track stages independently.

Example:

```text
Stage 1: discover files
Stage 2: validate inputs
Stage 3: decode
Stage 4: transform
Stage 5: generate embeddings
Stage 6: write database
Stage 7: verify outputs
```

For each stage track appropriate counters:

```text
input items
processed items
failed items
output items
throughput
elapsed time
```

Checkpoint expensive stages where practical.

Do not make a multi-hour preprocessing stage appear frozen.

---

## Simulations / experiments / sweeps

For independent trials, maintain per-trial status:

```text
pending
running
done
failed
```

Track:

```text
trial ID
parameters
seed
start time
completion time
result
failure
```

Store each trial independently when appropriate so completed trials do not need to rerun.

---

## Resource monitoring

For expensive jobs, periodically track resource health when practical.

### CPU

Useful values include:

```text
CPU utilization
load average
RAM usage
swap usage
worker count
```

### GPU

Useful values include:

```text
GPU utilization
GPU memory usage
temperature
power usage
```

### Storage

For I/O-heavy or data collection jobs:

```text
disk usage
output directory size
free disk space
read/write throughput when useful
```

### Network

For remote collection/download jobs:

```text
download throughput
request throughput
network failures
```

Monitoring must not materially degrade the workload.

---

## Stalled-job detection

A process existing does not necessarily mean the job is healthy.

Watch for situations such as:

```text
PID alive but no progress
GPU allocated but 0% utilization for unexpectedly long periods
CPU workers dead
download count no longer increasing
database writes stopped
repeated identical failures
disk full
OOM
NaN/Inf explosion
API authentication failure
rate-limit loop
```

The job log or progress state should update periodically enough to distinguish active work from a hung process.

When practical record:

```text
last_progress_at
```

---

## Checkpointing philosophy

A checkpoint means:

> enough persisted state exists to avoid unnecessarily repeating expensive completed work.

A checkpoint does not necessarily mean a model checkpoint.

Examples:

### Training

```text
model + optimizer + epoch
```

### Data collection

```text
completed item IDs + API cursor
```

### ETL

```text
completed shards + current stage
```

### Simulation

```text
completed trials
```

### Inference

```text
completed batches + predictions
```

### Database migration

```text
completed ranges/batches
```

Choose the checkpoint representation appropriate to the task.

---

## Completion markers

Use explicit run state where practical:

```text
RUNNING
DONE
FAILED
```

or:

```json
{
  "status": "completed"
}
```

`DONE` should only be written after required outputs have been successfully persisted.

A process disappearing is not itself proof of successful completion.

---

## Final summary

Every substantial job should produce a compact summary appropriate to the task.

Example for data collection:

```json
{
  "status": "completed",
  "total_items": 50000,
  "successful": 49871,
  "failed": 129,
  "bytes_downloaded": 89273192811,
  "runtime_seconds": 14281,
  "manifest": "manifest.csv",
  "failed_items": "failures.jsonl"
}
```

Example for training:

```json
{
  "status": "completed",
  "best_epoch": 27,
  "best_metric": "val_f1",
  "best_metric_value": 0.8421,
  "best_checkpoint": "checkpoints/best.pt",
  "last_checkpoint": "checkpoints/last.pt",
  "runtime_seconds": 8421
}
```

Adapt the summary to the actual task.

---

## Before reporting a job as healthy

Verify more than the PID.

Check:

1. the expected process is alive;
2. the workload has successfully initialized;
3. progress is actually occurring;
4. expected outputs are being written;
5. requested CPU/GPU resources are being respected;
6. logs show no fatal error;
7. progress/checkpoints are updating when applicable.

For data collection additionally verify a small sample of produced data where practical.

For training verify that metrics and checkpoints are appearing.

For database ingestion verify that committed row counts or output state are changing.

Do not report:

```text
"running successfully"
```

solely because a process exists.

Instead report concrete evidence, for example:

```text
Job launched successfully.

PID: 48122
Resources: all → 64 CPUs + 4 GPUs
Progress: 18,420 / 50,000 items (36.8%)
Rate: 4.8 items/sec
Failures: 37
Checkpoint: updated 14 seconds ago
Output: outputs/global_collection_20261002_170412/
Log: outputs/global_collection_20261002_170412/job.log
Status: process alive and progress increasing
```
