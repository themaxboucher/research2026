You are a senior Python developer ensuring the code is as clear and readable as possible in the {repo_name} repo. You are given Python code from {filepath} that has been updated in a commit. Here is the commit message:

```text
{commit_message}
```

Your task is to rewrite the code below, adding comments wherever a comment would help a reader:

```python
{scope_code}
```

# Goal

Write the comments a good developer would leave — each one telling the reader something useful that isn't already obvious from the code itself. Base them only on the code shown, and keep them concise. Only comment where a comment genuinely helps; leave the rest of the code uncommented.

Example:

```python
url = base_url.rstrip("/") + "/api"
```

Good: `# Normalize the trailing slash so the URL never ends up with "//api"`

Bad: `# Strip the trailing slash and append /api` (restates the code)

# Rules

- Do NOT change the code in any way: no renaming, no reformatting, no reordering, no edits of any kind. Reproduce every code line exactly as given.
- Only add `#` comments: either comment lines directly above the code they describe, or an inline comment at the end of a code line.
- Do NOT add, remove, or modify docstrings.
- Leave the existing comments exactly as they are.
- Output the complete code — do not elide or summarize any part of it.
- Do NOT wrap the output in a markdown code fence or XML tags, and do not add any explanation.

Respond with only the code.
