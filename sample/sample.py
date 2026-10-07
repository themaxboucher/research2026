import argparse
import hashlib
import json
import logging
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval.constants import BASELINE_MODELS, SCORE_METRICS, UNKNOWN_MODEL
from generate.constants import GENERATE_FILENAME
from storage import iter_from_jsonl, save_to_jsonl
from storage.runs import current_git_commit, resolve_dataset_and_run
from storage.timestamped_dirs import new_timestamped_directory

SAMPLES_DIRECTORY_NAME = "samples"
SAMPLE_FILENAME = "sample"

DEFAULT_METRIC = "bertscore_f1"
DEFAULT_FRACTION = 0.2
DEFAULT_SEED = 0


def _iter_scored_results(run_dir: Path, metric: str):
    """Yield (record, comment_generation, result, model, score) for every scored
    prediction of a real model. Eval leaves `scores` unset for errors and empty
    predictions or references, so those are skipped here too."""
    for record in iter_from_jsonl(run_dir, GENERATE_FILENAME + "_scored"):
        for comment_generation in record.get("comment_generations") or []:
            for result in comment_generation.get("results") or []:
                model = result.get("model") or UNKNOWN_MODEL
                if model in BASELINE_MODELS:
                    continue
                score = (result.get("scores") or {}).get(metric)
                if score is None:
                    continue
                yield record, comment_generation, result, model, score


def _nearest_rank_threshold(scores: list[float], fraction: float) -> float:
    rank = math.ceil(fraction * len(scores))
    return sorted(scores)[rank - 1]


def _item_key(record: dict, comment_generation: dict, model: str) -> dict:
    # The same fields eval uses to identify a result, so notes on a sampled
    # item can be joined back to its scores.
    return {
        "repo_name": record.get("repo_name"),
        "new_path": record.get("new_path"),
        "commit_hash": record.get("commit_hash"),
        "type": comment_generation.get("type"),
        "start_line": comment_generation.get("start_line"),
        "end_line": comment_generation.get("end_line"),
        "model": model,
    }


def _item_id(key: dict) -> str:
    serialized_key = json.dumps(list(key.values()), ensure_ascii=False)
    return hashlib.sha1(serialized_key.encode("utf-8")).hexdigest()[:16]


def _sample_item(
    record: dict, comment_generation: dict, result: dict, model: str, metric: str
) -> dict:
    key = _item_key(record, comment_generation, model)
    return {
        "id": _item_id(key),
        **key,
        "status": comment_generation.get("status"),
        "anchor": comment_generation.get("anchor"),
        "metric": metric,
        "score": result["scores"][metric],
        "scores": result["scores"],
        "reference_comment": comment_generation.get("comment"),
        "comment_text": result.get("comment_text"),
        "raw_response": result.get("raw_response"),
        "prompt_code": comment_generation.get("prompt_code"),
    }


def _scores_by_model(run_dir: Path, metric: str) -> dict[str, list[float]]:
    scores_by_model = defaultdict(list)
    for _, _, _, model, score in _iter_scored_results(run_dir, metric):
        scores_by_model[model].append(score)
    if not scores_by_model:
        raise SystemExit(f"No {metric} scores found in {run_dir.name}")
    return scores_by_model


def _check_candidate_counts(
    candidate_counts: dict[str, int], per_model: int
) -> None:
    short_models = {
        model: count for model, count in candidate_counts.items() if count < per_model
    }
    if short_models:
        details = ", ".join(
            f"{model} ({count})" for model, count in sorted(short_models.items())
        )
        raise SystemExit(
            f"Need {per_model} items per model but these models have fewer "
            f"below their cutoff: {details}"
        )


def _reservoir_sample(
    run_dir: Path,
    metric: str,
    thresholds: dict[str, float],
    per_model: int,
    rng: random.Random,
) -> list[dict]:
    """Draw `per_model` items uniformly from each model's candidates in one
    streaming pass, so the scored file never has to fit in memory."""
    reservoirs = defaultdict(list)
    candidates_seen = Counter()
    for record, comment_generation, result, model, score in _iter_scored_results(
        run_dir, metric
    ):
        if score > thresholds[model]:
            continue
        seen = candidates_seen[model]
        candidates_seen[model] += 1
        if seen < per_model:
            reservoirs[model].append(
                _sample_item(record, comment_generation, result, model, metric)
            )
            continue
        replace_index = rng.randrange(seen + 1)
        if replace_index < per_model:
            reservoirs[model][replace_index] = _sample_item(
                record, comment_generation, result, model, metric
            )

    items = [item for model in sorted(reservoirs) for item in reservoirs[model]]
    rng.shuffle(items)
    return items


def _write_manifest(
    sample_dir: Path,
    run_dir: Path,
    *,
    metric: str,
    fraction: float,
    seed: int,
    requested_total: int,
    per_model: int,
    scores_by_model: dict[str, list[float]],
    thresholds: dict[str, float],
    candidate_counts: dict[str, int],
    items: list[dict],
) -> None:
    models = {
        model: {
            "scored_count": len(scores_by_model[model]),
            "threshold": thresholds[model],
            "candidate_count": candidate_counts[model],
            "sampled_count": per_model,
        }
        for model in sorted(scores_by_model)
    }
    items_per_repo = Counter(item["repo_name"] for item in items)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": current_git_commit(),
        "run_dir": str(run_dir),
        "source_file": f"{GENERATE_FILENAME}_scored.jsonl",
        "metric": metric,
        "fraction": fraction,
        "seed": seed,
        "requested_total": requested_total,
        "per_model": per_model,
        "total": len(items),
        "models": models,
        "items_per_repo": dict(items_per_repo.most_common()),
    }
    (sample_dir / (SAMPLE_FILENAME + ".json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sample(run_dir: Path, *, total: int, metric: str, fraction: float, seed: int):
    scores_by_model = _scores_by_model(run_dir, metric)
    thresholds = {
        model: _nearest_rank_threshold(scores, fraction)
        for model, scores in scores_by_model.items()
    }
    candidate_counts = {
        model: sum(score <= thresholds[model] for score in scores)
        for model, scores in scores_by_model.items()
    }
    per_model = math.ceil(total / len(scores_by_model))
    _check_candidate_counts(candidate_counts, per_model)

    for model in sorted(scores_by_model):
        logging.info(
            "%s: %d scored, %s <= %.4f gives %d candidates",
            model,
            len(scores_by_model[model]),
            metric,
            thresholds[model],
            candidate_counts[model],
        )

    items = _reservoir_sample(
        run_dir, metric, thresholds, per_model, random.Random(seed)
    )

    sample_dir = new_timestamped_directory(run_dir / SAMPLES_DIRECTORY_NAME)
    save_to_jsonl(items, sample_dir, SAMPLE_FILENAME)
    _write_manifest(
        sample_dir,
        run_dir,
        metric=metric,
        fraction=fraction,
        seed=seed,
        requested_total=total,
        per_model=per_model,
        scores_by_model=scores_by_model,
        thresholds=thresholds,
        candidate_counts=candidate_counts,
        items=items,
    )
    logging.info(
        "Wrote %d items (%d per model) to %s", len(items), per_model, sample_dir
    )


def _fraction(value: str) -> float:
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("fraction must be in (0, 1]")
    return fraction


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to sample from (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to sample from (defaults to the latest run in the dataset)",
    )
    parser.add_argument(
        "--n",
        type=_positive_int,
        required=True,
        help="Total items to sample, split evenly across models and rounded up per model",
    )
    parser.add_argument(
        "--metric",
        choices=SCORE_METRICS,
        default=DEFAULT_METRIC,
        help=f"Score that ranks predictions (default: {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--fraction",
        type=_fraction,
        default=DEFAULT_FRACTION,
        help=(
            "Bottom fraction of each model's scores to sample from, using an "
            f"inclusive nearest-rank cutoff (default: {DEFAULT_FRACTION})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling and shuffling (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)
    _sample(
        run_directory,
        total=args.n,
        metric=args.metric,
        fraction=args.fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
