import evaluate

rouge = evaluate.load("rouge")

print(rouge.compute(
    predictions=["The quick brown fox jumps over the lazy dog."],
    references=["The quick brown fox jumps over a lazy dog."],
))