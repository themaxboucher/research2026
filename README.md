# Research 2026

Research project comparing **how humans write code comments** vs **how LLMs write code comments**, using GitHub commits made after each LLM's knowledge cutoff as the source of human-written ground truth.

## Research goal

1. Collect comments written by humans in high-quality public GitHub repos, **after** the knowledge cutoff of the LLMs we want to evaluate.
2. Quantify patterns in those comments (density per LOC, length, type, position, etc.).
3. Strip the comments from the same files, ask different LLMs to re-comment them, and compare the LLM output to the human ground truth.

Working with post-cutoff data avoids the risk that an LLM has memorized the file and its comments verbatim from training.
