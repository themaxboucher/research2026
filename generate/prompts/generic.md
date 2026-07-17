Write the comment a good developer would leave here — one that tells the reader something useful that isn't already obvious from the code itself. Base it only on the code shown, and keep it concise.

Example:

```python
url = base_url.rstrip("/") + "/api"
```

Good: `# Normalize the trailing slash so the URL never ends up with "//api"`

Bad: `# Strip the trailing slash and append /api` (restates the code)
