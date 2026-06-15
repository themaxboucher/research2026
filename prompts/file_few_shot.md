You are an expert Python developer reviewing a recent change to keep code comments accurate, useful, detailed, and concise.

A developer changed the Python file below in a commit. The file is shown in its post-change state, but its comments and docstrings have NOT been updated yet: comments may now be missing, outdated, or unnecessary. Your task is to make the comment and docstring edits that a thoughtful developer would make as part of this change. You may add, edit, and/or remove comments and docstrings.

<file path="{file_path}">
{code_file}
</file>

<change>
{diff}
</change>

# Comment types

- inline: a `#` comment at the end of an existing line of code.
- block: one or more `#`-only lines placed above the code they describe.
- docstring: a triple-quoted string that is the first statement of the module, a class, or a function/method.

# Editing rules

1. Base every edit on the change shown in <change>. Leave comments unrelated to this change untouched.
2. Edit only comments and docstrings. The code itself must remain exactly as shown.
3. Write the comments a thoughtful developer would write for this change: comment where the change makes existing comments wrong, where new behavior needs explanation, and nowhere else.
4. If the change requires no comment edits, respond with exactly: NO_EDITS

# Output format

Describe each change as a *SEARCH/REPLACE block*. Name the file, then give the exact lines to find and the lines to replace them with, using git-style conflict markers:

{file_path}
<<<<<<< SEARCH
a few lines copied verbatim from <file>, including the comments to change
=======
the same lines, with only the comments and docstrings corrected
>>>>>>> REPLACE

Rules for SEARCH/REPLACE blocks:

- The SEARCH section must reproduce a contiguous run of lines from <file> character-for-character, including indentation, blank lines, and the comments exactly as they currently appear.
- Keep the SEARCH section short, but include enough surrounding code that it matches exactly one location in the file.
- The REPLACE section must be identical to the SEARCH section except for its comments and docstrings. Every line of code must stay exactly the same.
- To add a comment, put the anchoring code in SEARCH and the new comment plus that code in REPLACE. To fix a comment, show the old comment in SEARCH and the corrected one in REPLACE. To remove a comment, include it in SEARCH and omit it from REPLACE.
- Use a separate SEARCH/REPLACE block for each region of the file you edit.

Respond with only the SEARCH/REPLACE blocks, or NO_EDITS, and no other text.

# Examples

Each example shows one edit. Your real response should contain only blocks like these.

## Adding

Add a docstring:

example.py
<<<<<<< SEARCH
def normalize(name):
    return name.strip().lower()
=======
def normalize(name):
    """Return name stripped of surrounding whitespace and lowercased."""
    return name.strip().lower()
>>>>>>> REPLACE

Add a block comment:

example.py
<<<<<<< SEARCH
    retries = 0
    while retries < MAX_RETRIES:
=======
    # Retry the request until it succeeds or we hit the limit.
    retries = 0
    while retries < MAX_RETRIES:
>>>>>>> REPLACE

Add an inline comment:

example.py
<<<<<<< SEARCH
    timeout = 30
=======
    timeout = 30  # seconds; long enough for slow CI hosts
>>>>>>> REPLACE

## Modifying

Modify a docstring:

example.py
<<<<<<< SEARCH
def load_config(path):
    """Load the config from a JSON file."""
    return tomllib.loads(Path(path).read_text())
=======
def load_config(path):
    """Load the config from a TOML file."""
    return tomllib.loads(Path(path).read_text())
>>>>>>> REPLACE

Modify a block comment:

example.py
<<<<<<< SEARCH
    # Fetch the user from the cache.
    user = db.get_user(user_id)
=======
    # Fetch the user from the database.
    user = db.get_user(user_id)
>>>>>>> REPLACE

Modify an inline comment:

example.py
<<<<<<< SEARCH
    buffer_size = 8192  # 4 KB buffer
=======
    buffer_size = 8192  # 8 KB buffer
>>>>>>> REPLACE

## Removing

Remove a docstring:

example.py
<<<<<<< SEARCH
def _slug(text):
    """TODO: document this helper."""
    return text.replace(" ", "-")
=======
def _slug(text):
    return text.replace(" ", "-")
>>>>>>> REPLACE

Remove a block comment:

example.py
<<<<<<< SEARCH
    # Legacy path: remove once everyone is on v2.
    response = client.send(payload)
=======
    response = client.send(payload)
>>>>>>> REPLACE

Remove an inline comment:

example.py
<<<<<<< SEARCH
    port = 8080  # default Flask port
=======
    port = 8080
>>>>>>> REPLACE
