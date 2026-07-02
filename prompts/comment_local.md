You are a senior Python developer keeping code comments relevant, accurate, useful, clear and concise.

A developer changed the Python code below in a commit, shown as a diff for the file at {file_path}. The comments have NOT been updated yet. Your task is to write the single most useful {comment_type} comment for the indicated location. Don't just summarize the code. The comment should tell a developer something useful that isn't already obvious from the code itself.

<change>
{diff_hunk}
</change>

# Location

{location_instruction}

# Rules

1. Base the comment on the change shown in <change> and the surrounding code.
2. Write the comment a senior developer would write for this change.
3. Output only the comment text — no code, no docstrings, no explanation.
4. For a block comment, output one or more lines, each beginning with `#`. For an inline comment, output a single `# ...`.

Respond with only the comment text.
