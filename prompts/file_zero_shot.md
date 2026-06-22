You are an expert Python developer reviewing a recent change to keep code comments accurate, useful, detailed, and concise.

A developer changed the Python file below in a commit. The file is shown in its post-change state, but its comments have NOT been updated yet: comments may now be missing, outdated, or unnecessary. Your task is to make the comment edits that a thoughtful developer would make as part of this change. You may add, edit, and/or remove comments.

<file path="{file_path}">
{code_file}
</file>

<change>
{diff}
</change>

# Comment types

- inline: a `#` comment at the end of an existing line of code.
- block: one or more `#`-only lines placed above the code they describe.

Only edit `#` comments. Do not add, edit, or remove docstrings (triple-quoted strings) or any other code.

# Editing rules

1. Base every edit on the change shown in <change>. Leave comments unrelated to this change untouched.
2. Edit only `#` comments. Docstrings and the code itself must remain exactly as shown.
3. Write the comments a thoughtful developer would write for this change: comment where the change makes existing comments wrong, where new behavior needs explanation, and nowhere else.
4. If the change requires no comment edits, respond with exactly: NO_EDITS

# Output format

Describe each change as a *SEARCH/REPLACE block*. Name the file, then give the exact lines to find and the lines to replace them with, using git-style conflict markers:

{file_path}
<<<<<<< SEARCH
a few lines copied verbatim from <file>, including the comments to change
=======
the same lines, with only the comments corrected
>>>>>>> REPLACE

Rules for SEARCH/REPLACE blocks:

- The SEARCH section must reproduce a contiguous run of lines from <file> character-for-character, including indentation, blank lines, and the comments exactly as they currently appear.
- Keep the SEARCH section short, but include enough surrounding code that it matches exactly one location in the file.
- The REPLACE section must be identical to the SEARCH section except for its comments. Every line of code, including docstrings, must stay exactly the same.
- To add a comment, put the anchoring code in SEARCH and the new comment plus that code in REPLACE. To fix a comment, show the old comment in SEARCH and the corrected one in REPLACE. To remove a comment, include it in SEARCH and omit it from REPLACE.
- Use a separate SEARCH/REPLACE block for each region of the file you edit.

Respond with only the SEARCH/REPLACE blocks, or NO_EDITS, and no other text.
