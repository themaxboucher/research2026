import argparse
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from generate.constants import SOURCE_FILENAME
from generate.providers.models import MODEL_PROFILES
from generate.providers.transformers import (
    MAX_OUTPUT_TOKENS,
    load_tokenizer,
    model_context_limit,
    prompt_token_count,
)
from storage.datasets import resolve_dataset_directory
from storage.jsonl import iter_from_jsonl, save_to_jsonl

TRANSFORMERS_PROFILE = "transformers"
DEFAULT_WORST_PROMPTS = 5
OUTPUT_FILENAME = "overflows.jsonl"


@dataclass
class ContextOverflow:
    model_name: str
    repo_name: str | None
    new_path: str | None
    commit_hash: str | None
    comment_type: str | None
    start_line: int | None
    end_line: int | None
    prompt_tokens: int
    overflow_tokens: int

    def comment_identity(self) -> tuple:
        return (
            self.repo_name,
            self.new_path,
            self.commit_hash,
            self.comment_type,
            self.start_line,
            self.end_line,
        )

    def as_record(self) -> dict:
        return {
            "model": self.model_name,
            "repo_name": self.repo_name,
            "new_path": self.new_path,
            "commit_hash": self.commit_hash,
            "type": self.comment_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "prompt_tokens": self.prompt_tokens,
            "overflow_tokens": self.overflow_tokens,
        }


@dataclass
class ModelContextTally:
    model_name: str
    context_limit: int
    budget_tokens: int
    comment_count: int = 0
    overflows: list[ContextOverflow] = field(default_factory=list)

    @property
    def overflow_count(self) -> int:
        return len(self.overflows)

    @property
    def overflow_percentage(self) -> float:
        if self.comment_count == 0:
            return 0.0
        return 100.0 * self.overflow_count / self.comment_count

    @property
    def file_count(self) -> int:
        return len({overflow.new_path for overflow in self.overflows})

    @property
    def median_overflow_tokens(self) -> int:
        if not self.overflows:
            return 0
        return round(
            statistics.median(overflow.overflow_tokens for overflow in self.overflows)
        )

    @property
    def worst_overflow_tokens(self) -> int:
        if not self.overflows:
            return 0
        return max(overflow.overflow_tokens for overflow in self.overflows)

    def worst_prompts(self, count: int) -> list[ContextOverflow]:
        return sorted(
            self.overflows, key=lambda overflow: overflow.prompt_tokens, reverse=True
        )[:count]


def _chat_template_overhead_tokens(tokenizer) -> int:
    return prompt_token_count(tokenizer, "")


def _fits_without_tokenizing(
    prompt: str, overhead_tokens: int, budget_tokens: int
) -> bool:
    """Decide a prompt fits without paying to tokenize it.

    No token spans more than one UTF-8 byte, so a prompt whose byte length plus
    the chat template's own tokens fits the budget cannot tokenize past it.
    """
    largest_possible_token_count = len(prompt.encode("utf-8")) + overhead_tokens
    return largest_possible_token_count <= budget_tokens


def _tally_target_comment(
    tally: ModelContextTally,
    tokenizer,
    overhead_tokens: int,
    record: dict,
    target_comment: dict,
) -> None:
    prompt = target_comment.get("prompt")
    if prompt is None:
        return

    tally.comment_count += 1
    if _fits_without_tokenizing(prompt, overhead_tokens, tally.budget_tokens):
        return

    prompt_tokens = prompt_token_count(tokenizer, prompt)
    if prompt_tokens <= tally.budget_tokens:
        return

    tally.overflows.append(
        ContextOverflow(
            model_name=tally.model_name,
            repo_name=record.get("repo_name"),
            new_path=record.get("new_path"),
            commit_hash=record.get("commit_hash"),
            comment_type=target_comment.get("type"),
            start_line=target_comment.get("start_line"),
            end_line=target_comment.get("end_line"),
            prompt_tokens=prompt_tokens,
            overflow_tokens=prompt_tokens - tally.budget_tokens,
        )
    )


def tally_context_overflows(
    dataset_directory: Path, model_name: str, reserve_tokens: int
) -> ModelContextTally:
    """Count the dataset's prompts that leave no room for `reserve_tokens` of output."""
    tokenizer = load_tokenizer(model_name)
    context_limit = model_context_limit(model_name)
    tally = ModelContextTally(
        model_name=model_name,
        context_limit=context_limit,
        budget_tokens=context_limit - reserve_tokens,
    )
    overhead_tokens = _chat_template_overhead_tokens(tokenizer)

    dataset_records = iter_from_jsonl(dataset_directory, SOURCE_FILENAME)
    for record in tqdm(dataset_records, desc=model_name, unit=" records"):
        for target_comment in record.get("target_comments") or []:
            _tally_target_comment(
                tally, tokenizer, overhead_tokens, record, target_comment
            )
    return tally


def _token_column(token_count: int) -> str:
    return f"{token_count:,}" if token_count else "–"


def _tally_row(tally: ModelContextTally, name_width: int) -> str:
    return (
        f"  {tally.model_name:<{name_width}}  {tally.budget_tokens:>9,}  "
        f"{tally.comment_count:>10,}  {tally.overflow_count:>10,}  "
        f"{tally.overflow_percentage:>5.1f}%  "
        f"{_token_column(tally.median_overflow_tokens):>11}  "
        f"{_token_column(tally.worst_overflow_tokens):>10}  {tally.file_count:>7,}"
    )


def _worst_prompt_lines(tally: ModelContextTally, worst_prompts: int) -> list[str]:
    if not tally.overflows or worst_prompts < 1:
        return []

    lines = ["", f"  Longest prompts for {tally.model_name}:"]
    for overflow in tally.worst_prompts(worst_prompts):
        lines.append(
            f"    {overflow.prompt_tokens:>9,} tokens "
            f"({overflow.overflow_tokens:,} over)   "
            f"{overflow.new_path}:{overflow.start_line}"
        )
    return lines


def _comments_over_limit_for_any_model(tallies: list[ModelContextTally]) -> int:
    return len(
        {
            overflow.comment_identity()
            for tally in tallies
            for overflow in tally.overflows
        }
    )


def _report_lines(
    dataset_directory: Path, tallies: list[ModelContextTally], worst_prompts: int
) -> list[str]:
    heading = (
        f"Prompts over the context limit in {dataset_directory.name} "
        f"({SOURCE_FILENAME}.jsonl)"
    )
    if not tallies:
        return [heading, "", "No models checked."]

    name_width = max(len(tally.model_name) for tally in tallies)
    header = (
        f"  {'Model':<{name_width}}  {'Budget':>9}  {'Comments':>10}  "
        f"{'Over limit':>10}  {'Share':>6}  {'Median over':>11}  "
        f"{'Worst over':>10}  {'Files':>7}"
    )
    lines = [heading, "", header]
    lines.extend(_tally_row(tally, name_width) for tally in tallies)

    for tally in tallies:
        lines.extend(_worst_prompt_lines(tally, worst_prompts))

    affected_comments = _comments_over_limit_for_any_model(tallies)
    checked_comments = max(tally.comment_count for tally in tallies)
    lines.append("")
    if affected_comments == 0:
        lines.append(f"Every prompt fits all {len(tallies)} models.")
        return lines

    affected_share = 100.0 * affected_comments / checked_comments
    lines.append(
        f"{affected_comments:,} of {checked_comments:,} comments ({affected_share:.1f}%) "
        f"exceed the limit of at least one model."
    )
    return lines


def print_context_report(
    dataset_directory: Path, tallies: list[ModelContextTally], worst_prompts: int
) -> None:
    print("\n".join(_report_lines(dataset_directory, tallies, worst_prompts)))


def write_overflow_records(tallies: list[ModelContextTally], output_path: Path) -> int:
    overflow_records = [
        overflow.as_record() for tally in tallies for overflow in tally.overflows
    ]
    save_to_jsonl(overflow_records, output_path.parent, output_path.stem)
    return len(overflow_records)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Report dataset prompts that exceed a model's context limit"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to check (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_names",
        default=None,
        help="Model to check. Repeatable; defaults to every model in the "
        "transformers profile",
    )
    parser.add_argument(
        "--reserve-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=f"Output tokens to leave room for (default: {MAX_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--worst-prompts",
        type=int,
        default=DEFAULT_WORST_PROMPTS,
        help=f"Longest prompts to list per model (default: {DEFAULT_WORST_PROMPTS})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUTPUT_FILENAME,
        help=f"Write the overflowing comments to this .jsonl file (default: {OUTPUT_FILENAME})",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = resolve_dataset_directory(args.dataset_dir)
    if not (dataset_directory / f"{SOURCE_FILENAME}.jsonl").exists():
        raise SystemExit(f"No {SOURCE_FILENAME}.jsonl in {dataset_directory}.")

    model_names = args.model_names or MODEL_PROFILES[TRANSFORMERS_PROFILE].model_names
    tallies = [
        tally_context_overflows(dataset_directory, model_name, args.reserve_tokens)
        for model_name in model_names
    ]

    print_context_report(dataset_directory, tallies, args.worst_prompts)

    if args.out:
        output_path = Path(args.out)
        written = write_overflow_records(tallies, output_path)
        logging.info("Wrote %d overflowing comments to %s", written, output_path)


if __name__ == "__main__":
    main()
