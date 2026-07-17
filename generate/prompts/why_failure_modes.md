# Wrong rationale

The comment's stated rationale is contradicted by the context.

Example:

```python
data = fetch_from_cache(key)
```

`# Hit the network here because the cache is unreliable` (the code reads from the cache, not the network)

# Unsupported rationale

The comment's stated rationale is plausible but not directly supported by the context. It is invented or guessed.

Example:

```python
items = list(self.items)
```

`# Copy the list because other threads may mutate it while we iterate` (nothing in the context mentions threads)

# Generic rationale

The comment's stated rationale is trivial and offers no insight that isn't already obvious from the code.

Example:

```python
for user in users:
    notify(user)
```

`# Loop over the users so each one gets notified`

# No rationale

The comment does not offer a rationale for the code's implementation. It is not a _why_ comment.

Example:

```python
for conn in list(self.pool):
    if conn.is_stale():
        self.pool.remove(conn)
```

`# Remove stale connections from the pool` (describes what the loop does, not why it iterates over a copy)
