You are a senior Python developer keeping code comments relevant, accurate, useful, clear and concise.

A developer changed the Python code below in a commit, shown as a diff for the file at {filepath}. The comments have NOT been updated yet. Your task is to edit or add to the following comment:

```python
{unmodified_comment}
```

<change>
{diff_hunk}
</change>

# Rules

1. Base the comment on the change shown in <change> and the surrounding code.
2. Write the comment a senior developer would write for this change.
4. Output only the comment text — no code, no docstrings, no explanation.
5. For a block comment, output one or more lines, each beginning with `#`. For an inline comment, output a single `# ...`.
6. Do NOT wrap the comment in a markdown code fence or XML tags.

Respond with only the comment text.
