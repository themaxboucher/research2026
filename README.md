# Research 2026

An experiment on how well LLMs write Python code comments: we mine real
comments from GitHub commits, ask models to reproduce them, and score the
output against the human-written originals.

It runs in three stages: **collect**, **generate**, **eval**. Each shares a
dataset directory under `datasets/<timestamp>/` and default to the latest one.
The main way to run it is as SLURM job arrays on an HPC cluster.

## Setup

Add secrets to `.env` (GitHub token(s), plus `OPENROUTER_API_KEY` / `HF_TOKEN`
for generate). The submit scripts create the `.venv` and install deps for you.

## Running on the cluster

```bash
./collect/scripts/submit.sh         # 1. mine repos into a dataset
./collect/scripts/filter-submit.sh  #    filter it down to the files worth generating on
./generate/scripts/submit.sh        # 2. query the LLMs for comment predictions
./eval/scripts/submit.sh            # 3. score predictions (BLEU / ROUGE / BERTScore)
```

Each `submit.sh` runs a prepare step, submits the mining/generation job array,
then a finalize job to merge the shards. Pass `--help` for options (`--dataset-dir`
/ `--run-dir` to resume, `--array` to rerun tasks, throttles, limits).

The filter step is a separate submission rather than part of
`collect/scripts/submit.sh`, so it has to be run once the mining array's finalize
job has merged the shards. It filters as a job array too followed by a dependent job
that merges the shards and their manifests.

The filter's last step drops every target comment whose prompt leaves no room
for the output in the context of any model in the `transformers` profile, each
counted with its own tokenizer. A dataset filtered before that step can be
brought in line without refiltering or regenerating. Back up `dataset.jsonl`,
`manifest.json` and each run's `generated.jsonl` / `generated_scored.jsonl` (both
scripts rewrite them in place), then:

```bash
sbatch collect/scripts/filter-context-overflow.sh  # drop the overflows from the dataset and update its manifest
sbatch generate/scripts/prune.sh                   # drop the run's generations for comments no longer in the dataset
```

Then rerun `eval.finalize --skip-merge` and the analyses. A pruned run is
finished: don't resume, retry or re-finalize its generation, since its shards
still hold the pruned comments.

Generation defaults to local inference on the cluster's GPUs (the `transformers`
backend). Pass `--profile openrouter` to route inference through the OpenRouter
API instead — no GPU is used, but the compute nodes need outbound internet and
`OPENROUTER_API_KEY`.

Once a run finishes, `./collect/scripts/pull-datasets.sh` rsyncs the dataset
back to your machine (configure `CLUSTER_SSH_HOST` / `CLUSTER_REMOTE_DIR` in `.env`).
