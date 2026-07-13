Your task is to classify a code comment based on its intent. The possible categories are as follows.

# Why

The comment explains why the code is implemented the way it is — the rationale behind it. It might point out an edge case being handled, a bug being prevented, a performance consideration, or a domain constraint or external requirement. It tells the reader something that isn't obvious from the code itself.

Example: `# Iterate over a copy — removing from the live list mid-loop would skip elements`

# What

The comment summarizes what the code does — its purpose or outcome — without going into the mechanics of how it works or the reasoning behind it. It helps a reader grasp the code at a glance.

Example: `# Turn a title into a URL-safe slug`

# How

The comment explains how the code works — the core logic, algorithm, or key steps it takes to get its result. It walks through non-obvious mechanics the reader would otherwise have to trace by hand.

Example: `# Floyd's tortoise-and-hare: a fast pointer laps the slow one when the list loops`

# Other

The comment doesn't explain the code at all. This includes TODOs and FIXMEs, commented-out code, license headers, section dividers, directives for tools (e.g. `# noqa`, `# type: ignore`), and notes to other developers unrelated to the code's behavior.

Example: `# TODO: add retry logic once the new client ships`

Respond with only Why, What, How or Other. No other text.
