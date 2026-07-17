Explain **how** the code works — the core logic, algorithm, or key steps it takes to get its result. Focus on non-obvious mechanics a reader would otherwise have to trace by hand; don't paraphrase the code line by line. Base the explanation only on the code shown, and keep it as concise as the logic allows.

Example:
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

Good: `# Floyd's tortoise-and-hare: a fast pointer laps the slow one when the list loops`

Bad: `# Checks whether the linked list has a cycle` (that's what, not how)
