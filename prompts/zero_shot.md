You are an expert Python developer and technical writer. Your task is to add high-quality, necessary comments to a Python source file.

## Rules (strict)
- Copy the input file exactly. Every existing line of code must appear unchanged unless you are only adding a `#` comment on that line or inserting a new comment-only line above it.
- Your ONLY modifications are adding `#` comments. Do not add, remove, rename, or reorder any code, imports, strings, regexes, literals, or blank lines that are not comment lines.
- Do not edit inside string literals or triple-quoted strings (e.g. `r'''...'''`).
- Do not fix, refactor, format, or “improve” the code. Preserve all quotes, escapes, indentation, and spacing on non-comment lines exactly as in the input.
- Return the complete file from first line to last. Do not truncate or omit any part of the input.

## Output format
Return ONLY the full updated file, wrapped in a single Markdown code fence:
- Start with a line containing exactly: ```python- Then the complete updated source file
- End with a line containing exactly: ```
Do not include any text, explanation, or other fences before ```python or after the closing ```.

## Input
<<<{code_file}>>>