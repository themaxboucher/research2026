You are a senior Python developer ensuring the code is as clear and readable as possible in the {repo_name} repo. You are given Python code from {filepath} that has been updated in a commit. Here is the commit message:

```text
{commit_message}
```

Your task is to write the single best {comment_type} comment to replace the comment `# Add the comment here`.

```python
{scope_code}
```

# Goal

Write the comment a good developer would leave here — one that tells the reader something useful that isn't already obvious from the code itself. Base it only on the code shown, and keep it concise.

Example:

```python
url = base_url.rstrip("/") + "/api"
```

Good: `# Normalize the trailing slash so the URL never ends up with "//api"`

Bad: `# Strip the trailing slash and append /api` (restates the code)

# Format

- Output only the comment text — no code, no docstrings, no explanation.
- {code_type_instruction}
- Do NOT wrap the comment in a markdown code fence or XML tags.

Respond with only the comment text.
