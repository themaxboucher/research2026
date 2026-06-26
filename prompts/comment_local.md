You are an expert Python developer keeping code comments accurate, useful, detailed, and concise.

A developer changed the Python code below in a commit. The relevant scope is shown in its post-change state, but its comments have NOT been updated yet. The code change is shown as a diff. Your task is to write the single {comment_type} comment that belongs at the indicated location.

<scope path="{file_path}">
{scope_code}
</scope>

<change>
{scope_diff}
</change>

# Location

{location_instruction}

# Rules

1. Base the comment on the change shown in <change> and the surrounding code.
2. Write the comment a thoughtful developer would write for this change.
3. Output only the comment text — no code, no docstrings, no explanation.
4. For a block comment, output one or more lines, each beginning with `#`. For an inline comment, output a single `# ...`.
5. Do not indent the comment; indentation is applied automatically.

Respond with only the comment text.
