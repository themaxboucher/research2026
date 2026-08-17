import evaluate
import torch
from bert_score import BERTScorer
from tqdm.auto import tqdm

from eval.constants import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_CHUNK_SIZE,
    BERTSCORE_MAX_LENGTH,
    BERTSCORE_MODEL,
    BLEU_MAX_ORDER,
)


class CommentScorer:
    def __init__(self):
        self._bleu = None
        self._rouge = None
        self._bertscore = None

    def _load(self):
        # Load the metric models if they haven't been loaded yet.
        # We load them lazily so that runs with no results to score don't pay the cost of loading BERTScore.
        if self._rouge is None or self._bertscore is None:
            self._rouge = evaluate.load("rouge", keep_in_memory=True)
            # BERTScore is the expensive metric, so run it on the GPU when one is
            # present and fall back to CPU otherwise (e.g. local runs).
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use BERTScorer directly (not evaluate's wrapper) so we can reach the
            # tokenizer and clamp its overflowing model_max_length.
            self._bertscore = BERTScorer(model_type=BERTSCORE_MODEL, device=device)
            self._bertscore._tokenizer.model_max_length = BERTSCORE_MAX_LENGTH

    def _load_bleu(self):
        # BLEU loads separately because corpus_bleu is needed at aggregate
        # time even when there are no new pairs to score.
        if self._bleu is None:
            self._bleu = evaluate.load("bleu", keep_in_memory=True)

    def corpus_bleu(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        self._load_bleu()
        return {
            f"bleu{order}_corpus": self._bleu.compute(
                predictions=predictions, references=references, max_order=order
            )["bleu"]
            for order in range(1, BLEU_MAX_ORDER + 1)
        }

    def score_pairs(
        self,
        predictions: list[str],
        references: list[str],
        desc: str = "Scoring",
    ) -> list[dict]:
        self._load()
        self._load_bleu()

        # Score in chunks so BERTScore can run forward passes in parallel
        scores = []
        with tqdm(total=len(predictions), desc=desc, unit="pair") as progress_bar:
            for start in range(0, len(predictions), BERTSCORE_CHUNK_SIZE):
                pred_chunk = predictions[start : start + BERTSCORE_CHUNK_SIZE]
                ref_chunk = references[start : start + BERTSCORE_CHUNK_SIZE]

                bleu_scores = [
                    {
                        f"bleu{order}": self._bleu.compute(
                            predictions=[pred], references=[ref], max_order=order
                        )["bleu"]
                        for order in range(1, BLEU_MAX_ORDER + 1)
                    }
                    for pred, ref in zip(pred_chunk, ref_chunk)
                ]
                rouge_scores = self._rouge.compute(
                    predictions=pred_chunk,
                    references=ref_chunk,
                    rouge_types=["rougeL"],
                    use_aggregator=False,
                )["rougeL"]
                # BERTScorer.score returns (P, R, F) tensors. We keep F1.
                # It re-batches the chunk internally for the GPU, so batch_size
                # bounds peak memory independently of how large a chunk we feed.
                bertscore_f1 = self._bertscore.score(
                    pred_chunk, ref_chunk, batch_size=BERTSCORE_BATCH_SIZE
                )[2].tolist()

                scores.extend(
                    {**bleu, "rougeL": rouge, "bertscore_f1": bert}
                    for bleu, rouge, bert in zip(
                        bleu_scores, rouge_scores, bertscore_f1
                    )
                )
                progress_bar.update(len(pred_chunk))

        return scores
