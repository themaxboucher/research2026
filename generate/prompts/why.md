Don't just summarize the code. The comment should explain **why** the code is implemented as it is. Maybe this implementation accounts for an edge case, prevents a bug, makes the code more efficient, or encodes a domain constraint or external requirement. The goal is to tell a developer something useful that isn't already obvious from the code itself. The rationale should be supported by the code. Do NOT invent or guess the rationale. If the code doesn't reveal a non-obvious reason, prefer a short accurate note over a speculative one. Ensure the comment is clear and easily understandable. The comment should be as concise as possible without sacrificing important details.

Example:
```python
for conn in list(self.pool):
    if conn.is_stale():
        self.pool.remove(conn)
```

Good: `# Iterate over a copy — removing from the live list mid-loop would skip the element after each removal`

Bad: `# Remove stale connections from the pool` (that's what the loop does; it doesn't explain the otherwise-pointless list() copy)
