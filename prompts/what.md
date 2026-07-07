Summarize **what** the referenced code does — its purpose or outcome — not how or why. Help a reader grasp complex logic at a glance. Skip obvious descriptions of simple functionality. Base the summary only on the code shown. Be clear, concise, and cut filler: instead of "This line of code does x," just say "x."

Example:
```python
def _slugify(title):
    cleaned = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_-]+", "-", cleaned).strip("-")
```

Good: `# Turn a title into a URL-safe slug`

Bad: `# Lowercase the title, strip punctuation, then replace whitespace with dashes` (that's how, not what)
