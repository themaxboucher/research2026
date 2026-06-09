You are an expert Python developer and technical writer. A developer has just changed a Python file in a commit. Your task is to add the comments and docstrings you would write for this change to the NEW version of the file.
{previous_section}

## New version of the file (comments and docstrings removed)

This is the file after the change, with all comments and docstrings stripped. Add the comments and docstrings you would write when making this change.
<<<{code_file}>>>

## What you may add

- Inline comments: a `#` comment at the end of an existing line of code.
- Block comments: one or more `#` comment-only lines inserted above the code they describe.
- Docstrings: a triple-quoted string as the FIRST statement of the module, a class, or a function/method.

## Rules (strict)

- Copy the input file exactly. Every existing line of code must appear unchanged unless you are adding a `#` comment on that line, inserting a new comment-only line, or inserting a new docstring.
- Your ONLY modifications are adding `#` comments and docstrings. Do not add, remove, rename, or reorder any code, imports, strings, regexes, literals, or blank lines that are not comment or docstring lines.
- A docstring may only be inserted as the first statement of the module, a class body, or a function/method body. Do not otherwise edit or wrap existing string literals.
- Do not fix, refactor, format, or “improve” the code. Preserve all quotes, escapes, indentation, and spacing on non-comment lines exactly as in the input.
- Comment only what a thoughtful developer would comment for this change. Do not add a comment to every line, and do not add a docstring to every function.
- Return the complete file from first line to last. Do not truncate or omit any part of the input.

## Output format

Return ONLY the full updated file, wrapped in a single Markdown code fence:

- Start with a line containing exactly: ```python
- Then the complete updated source file
- End with a line containing exactly: `Do not include any text, explanation, or other fences before`python or after the closing ```.
