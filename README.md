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
./collect/scripts/submit.sh      # 1. mine repos into a dataset
python -m collect.sample         #    sample the commits to run on
./generate/scripts/submit.sh     # 2. query the LLMs for comment predictions
python -m eval.eval              # 3. score predictions (BLEU / ROUGE / BERTScore)
```

Each `submit.sh` runs a prepare step, submits the mining/generation job array,
then a finalize job to merge the shards. Pass `--help` for options (`--dataset-dir`
/ `--run-dir` to resume, `--array` to rerun tasks, throttles, limits).

Once a run finishes, `./collect/scripts/pull-datasets.sh` rsyncs the dataset
back to your machine (configure `CLUSTER_SSH_HOST` / `CLUSTER_REMOTE_DIR` in `.env`).

## Running locally

Each stage is also a plain Python package you can invoke directly — see the
`python -m ...prepare / ...collect / ...generate / ...finalize` steps inside the
submit scripts, run in that order.
