# These are the approaches we use to prompt the LLMs. Different approaches produce
# different types of outputs (e.g. generate only the comment vs regenerate the
# whole code with comments added).
APPROACHES = ("location", "regenerate")

def approaches_from_argument(approaches_arg: str | None) -> list[str]:
    approaches = [
        approach.strip() for approach in approaches_arg.split(",") if approach.strip()
    ]
    approaches = list(approaches or APPROACHES)
    unknown_approaches = set(approaches) - set(APPROACHES)
    if unknown_approaches:
        raise ValueError(
            f"Unknown approaches: {', '.join(sorted(unknown_approaches))}. "
            f"Expected any of: {', '.join(APPROACHES)}"
        )
    return approaches