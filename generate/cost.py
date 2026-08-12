import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from generate.constants import GENERATE_FILENAME
from storage.jsonl import iter_from_jsonl
from storage.runs import resolve_dataset_and_run

DEFAULT_TOKENIZER_MODEL = "gpt-5.6-luna"
GPT_5_FAMILY_ENCODING = "o200k_base"
TOKENIZER_THREAD_COUNT = 8
PROGRESS_LOG_INTERVAL_RECORDS = 20000
TOKENS_PER_PRICE_UNIT = 1_000_000


def resolve_encoding(tokenizer_model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(tokenizer_model)
    except KeyError:
        # tiktoken has no model mapping for the newest GPT-5 releases yet, but
        # the whole GPT-5 family shares the o200k_base vocabulary.
        logging.warning(
            "tiktoken has no mapping for %r; falling back to the %s encoding",
            tokenizer_model,
            GPT_5_FAMILY_ENCODING,
        )
        return tiktoken.get_encoding(GPT_5_FAMILY_ENCODING)


def count_tokens_per_text(encoding: tiktoken.Encoding, texts: list[str]) -> list[int]:
    # encode_ordinary treats special-token markers appearing in source code or
    # model output as ordinary text instead of rejecting them.
    token_lists = encoding.encode_ordinary_batch(
        texts, num_threads=TOKENIZER_THREAD_COUNT
    )
    return [len(tokens) for tokens in token_lists]


@dataclass
class TokenTotals:
    total_tokens: int = 0
    sample_count: int = 0

    def add(self, token_count: int) -> None:
        self.total_tokens += token_count
        self.sample_count += 1

    @property
    def average_tokens(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.total_tokens / self.sample_count


@dataclass
class RunTokenStats:
    prompts: TokenTotals = field(default_factory=TokenTotals)
    outputs: TokenTotals = field(default_factory=TokenTotals)
    outputs_by_model: dict[str, TokenTotals] = field(
        default_factory=lambda: defaultdict(TokenTotals)
    )
    failed_result_count: int = 0


@dataclass
class RecordTexts:
    prompts: list[str]
    model_outputs: list[tuple[str, str]]
    failed_result_count: int


def _extract_record_texts(record: dict) -> RecordTexts:
    """Split one file record into the prompt texts and the per-model output texts
    that should be tokenized. Results whose generation call failed carry no
    response text and are counted as failures instead."""
    prompts = []
    model_outputs = []
    failed_result_count = 0

    for generation in record["comment_generations"]:
        prompts.append(generation["prompt"])
        for result in generation["results"]:
            if result["raw_response"] is None:
                failed_result_count += 1
                continue
            model_outputs.append((result["model"], result["raw_response"]))

    return RecordTexts(prompts, model_outputs, failed_result_count)


def _add_record_to_stats(
    stats: RunTokenStats, record_texts: RecordTexts, encoding: tiktoken.Encoding
) -> None:
    stats.failed_result_count += record_texts.failed_result_count

    for token_count in count_tokens_per_text(encoding, record_texts.prompts):
        stats.prompts.add(token_count)

    output_texts = [text for _, text in record_texts.model_outputs]
    output_token_counts = count_tokens_per_text(encoding, output_texts)
    for (model_name, _), token_count in zip(
        record_texts.model_outputs, output_token_counts
    ):
        stats.outputs.add(token_count)
        stats.outputs_by_model[model_name].add(token_count)


def collect_token_stats(
    run_directory: Path, tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
) -> RunTokenStats:
    """Count prompt and model-output tokens across every generation in a run."""
    encoding = resolve_encoding(tokenizer_model)
    stats = RunTokenStats()

    for record_number, record in enumerate(
        iter_from_jsonl(run_directory, GENERATE_FILENAME), start=1
    ):
        _add_record_to_stats(stats, _extract_record_texts(record), encoding)
        if record_number % PROGRESS_LOG_INTERVAL_RECORDS == 0:
            logging.info("Tokenized %d records", record_number)

    return stats


@dataclass
class TokenPricing:
    input_price_per_million: float
    output_price_per_million: float


@dataclass
class CostEstimate:
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


def estimate_cost(stats: RunTokenStats, pricing: TokenPricing) -> CostEstimate:
    """Project what the run would cost at the given prices, charging one
    completion of average length for every prompt in the run."""
    prompt_count = stats.prompts.sample_count
    billed_input_tokens = stats.prompts.average_tokens * prompt_count
    billed_output_tokens = stats.outputs.average_tokens * prompt_count

    return CostEstimate(
        input_cost=billed_input_tokens
        / TOKENS_PER_PRICE_UNIT
        * pricing.input_price_per_million,
        output_cost=billed_output_tokens
        / TOKENS_PER_PRICE_UNIT
        * pricing.output_price_per_million,
    )


def _log_cost_estimate(stats: RunTokenStats, pricing: TokenPricing) -> None:
    cost = estimate_cost(stats, pricing)
    logging.info(
        "Cost for %d prompts at $%s/1M input and $%s/1M output tokens, "
        "billing one average-length completion per prompt:",
        stats.prompts.sample_count,
        pricing.input_price_per_million,
        pricing.output_price_per_million,
    )
    logging.info("  Input cost:  $%s", f"{cost.input_cost:,.2f}")
    logging.info("  Output cost: $%s", f"{cost.output_cost:,.2f}")
    logging.info("  Total cost:  $%s", f"{cost.total_cost:,.2f}")


def _log_token_stats(run_directory: Path, stats: RunTokenStats) -> None:
    logging.info(
        "%s: %d prompts, %d model outputs, %d failed generations excluded",
        run_directory.name,
        stats.prompts.sample_count,
        stats.outputs.sample_count,
        stats.failed_result_count,
    )
    logging.info(
        "Average prompt tokens: %.1f (%d total)",
        stats.prompts.average_tokens,
        stats.prompts.total_tokens,
    )
    logging.info(
        "Average output tokens: %.1f (%d total)",
        stats.outputs.average_tokens,
        stats.outputs.total_tokens,
    )
    for model_name in sorted(stats.outputs_by_model):
        model_outputs = stats.outputs_by_model[model_name]
        logging.info(
            "Average output tokens for %s: %.1f (%d outputs)",
            model_name,
            model_outputs.average_tokens,
            model_outputs.sample_count,
        )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to report on (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to report on (defaults to the latest run in the dataset)",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=DEFAULT_TOKENIZER_MODEL,
        help=f"Model whose tiktoken encoding is used (defaults to {DEFAULT_TOKENIZER_MODEL})",
    )
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        default=None,
        help="Price in dollars per 1M input tokens; enables the cost estimate",
    )
    parser.add_argument(
        "--output-price-per-million",
        type=float,
        default=None,
        help="Price in dollars per 1M output tokens; enables the cost estimate",
    )
    return parser.parse_args()


def _pricing_from_args(args) -> TokenPricing | None:
    if args.input_price_per_million is None or args.output_price_per_million is None:
        return None
    return TokenPricing(args.input_price_per_million, args.output_price_per_million)


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)

    stats = collect_token_stats(run_directory, args.tokenizer_model)
    _log_token_stats(run_directory, stats)

    pricing = _pricing_from_args(args)
    if pricing is None:
        logging.info(
            "No cost estimate: pass --input-price-per-million and "
            "--output-price-per-million to price this run"
        )
        return

    _log_cost_estimate(stats, pricing)


if __name__ == "__main__":
    main()
