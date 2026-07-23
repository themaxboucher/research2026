import evaluate
from bert_score import BERTScorer
from tqdm.auto import tqdm

from eval.constants import (
    BERTSCORE_MAX_LENGTH,
    BERTSCORE_MODEL,
)


class CommentScorer:
    """Computes the three metrics over batches of (prediction, reference) pairs."""

    def __init__(self):
        self._bleu = None
        self._rouge = None
        self._bertscore = None

    def _load(self):
        # Load the metric models if they haven't been loaded yet.
        # We load them lazily so that runs with no results to score don't pay the cost of loading BERTScore.
        if self._rouge is None or self._bertscore is None:
            self._rouge = evaluate.load("rouge")
            # Use BERTScorer directly (not evaluate's wrapper) so we can reach the
            # tokenizer and clamp its overflowing model_max_length.
            self._bertscore = BERTScorer(model_type=BERTSCORE_MODEL)
            self._bertscore._tokenizer.model_max_length = BERTSCORE_MAX_LENGTH

    def _load_bleu(self):
        # BLEU loads separately because corpus_bleu is needed at aggregate
        # time even when there are no new pairs to score.
        if self._bleu is None:
            self._bleu = evaluate.load("bleu")

    def corpus_bleu(self, predictions: list[str], references: list[str]) -> float:
        """Standard (unsmoothed) corpus-level BLEU-4 over all pairs at once."""
        self._load_bleu()
        return self._bleu.compute(
            predictions=predictions, references=references, max_order=4
        )["bleu"]

    def score_pairs(
        self,
        predictions: list[str],
        references: list[str],
        desc: str = "Scoring",
    ) -> list[dict]:
        self._load()
        self._load_bleu()

        # Score in batches so BERTScore can run forward passes in parallel
        scores = []
        BATCH_SIZE = 64
        with tqdm(total=len(predictions), desc=desc, unit="pair") as progress_bar:
            for start in range(0, len(predictions), BATCH_SIZE):
                pred_batch = predictions[start : start + BATCH_SIZE]
                ref_batch = references[start : start + BATCH_SIZE]

                # We use per comment pair BLEU 4 here (not corpus-level).
                # Smoothing is needed, otherwise any pair without a matching
                # 4-gram scores 0.
                bleu_scores = [
                    self._bleu.compute(
                        predictions=[pred], references=[ref], max_order=4, smooth=True
                    )["bleu"]
                    for pred, ref in zip(pred_batch, ref_batch)
                ]
                rouge_scores = self._rouge.compute(
                    predictions=pred_batch,
                    references=ref_batch,
                    rouge_types=["rougeL"],
                    use_aggregator=False,
                )["rougeL"]
                # BERTScorer.score returns (P, R, F) tensors. We keep F1.
                bertscore_f1 = self._bertscore.score(pred_batch, ref_batch)[2].tolist()

                scores.extend(
                    {"bleu4": bleu, "rougeL": rouge, "bertscore_f1": bert}
                    for bleu, rouge, bert in zip(
                        bleu_scores, rouge_scores, bertscore_f1
                    )
                )
                progress_bar.update(len(pred_batch))

        return scores
