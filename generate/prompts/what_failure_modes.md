# Inaccurate summary

The comment says the code does something it does not do.

Example:

```python
results.sort(key=lambda r: r.created_at)
```

`# Sort results from newest to oldest` (the sort is ascending, so oldest comes first)

# Confusing summary

The comment does not make the code easier to understand. It is vague, ambiguous, or otherwise unhelpful.

Example:

```python
orders = [o for o in orders if o.total >= MIN_TOTAL]
```

`# Handle the orders according to the relevant threshold` (says nothing concrete about what happens)

# Wrong scope summary

The comment summarizes too much or too little of the code, or a different part of the code than it is attached to.

Example:

```python
user = db.get_user(user_id)  # Fetch the user and send them a welcome email
send_welcome_email(user)
```

(the comment on the first line describes both lines)

# No summary

The comment is not a summary of the code. It is not a _what_ comment.

Example:

```python
total = sum(item.price for item in cart)
```

`# TODO: handle discounts later`
