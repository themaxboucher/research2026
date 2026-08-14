import ast

from cognitive_complexity.api import get_cognitive_complexity_for_node
from radon.visitors import ComplexityVisitor

STRAIGHT_LINE = '''
def area(width, height):
    """No branches at all: the baseline for both metrics."""
    return width * height
'''

FLAT_CHAIN = '''
def grade(score):
    """Four branches, but flat, so a reader only holds one level in mind."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
'''

NESTED_CONDITIONS = '''
def discount(customer, cart, coupon):
    """The same number of branches as grade(), but stacked three deep."""
    if customer.is_member:
        if cart.total > 100:
            if coupon is not None:
                return 0.3
            return 0.2
        return 0.1
    return 0.0
'''

BOOLEAN_OPERATORS = '''
def is_eligible(user):
    """Cyclomatic counts each `and`/`or`; cognitive counts each run of them."""
    return user.is_active and user.age >= 18 and user.country == "CA"
'''

NESTED_LOOP = '''
def sum_of_primes(limit):
    """Loops nest the same way conditions do, and the inner `if` pays for both."""
    total = 0
    for number in range(2, limit):
        for divisor in range(2, number):
            if number % divisor == 0:
                break
        else:
            total += number
    return total
'''

CLASS_WITH_METHODS = '''
class Cart:
    """A class is not a unit of complexity: the score is what its methods add
    up to, and a class of four small methods stays cheap."""

    def __init__(self, items=None):
        self.items = items or []

    @property
    def total(self):
        return sum(item.price * item.quantity for item in self.items)

    def add(self, item):
        for existing in self.items:
            if existing.sku == item.sku:
                existing.quantity += item.quantity
                return
        self.items.append(item)

    @staticmethod
    def shipping(total, country):
        if country != "CA":
            return 25.0
        return 0.0 if total > 50 else 9.99
'''

CLOSURE = '''
def make_rate_limiter(limit, window):
    """radon scores the closure as its own block and then leaves it out of the
    module total; cognitive complexity folds it into the enclosing function and
    charges the branches inside it an extra nesting level."""
    calls = []

    def allow(now):
        while calls and now - calls[0] > window:
            calls.pop(0)
        if len(calls) >= limit:
            return False
        calls.append(now)
        return True

    return allow
'''

COMPREHENSIONS = '''
def summarize(records):
    """Cyclomatic charges every comprehension plus each `if` inside it;
    cognitive ignores comprehensions and only sees the ternary."""
    names = [r.name for r in records if r.active]
    by_owner = {r.owner: r for r in records if r.owner is not None}
    flagged = {r.id for r in records for tag in r.tags if tag == "flag"}
    label = "empty" if not names else "populated"
    return names, by_owner, flagged, label
'''

ERROR_HANDLING = '''
def load_config(path, parser, default):
    """Each handler costs 1 in both metrics, `else` costs 1 more in
    cyclomatic, and `finally` is free in both."""
    try:
        raw = open(path).read()
    except FileNotFoundError:
        return default
    except PermissionError as exc:
        if exc.errno == 13:
            raise RuntimeError("config is unreadable") from exc
        return default
    else:
        config = parser(raw)
    finally:
        print("config load attempted")
    return config
'''

RECURSION = '''
def flatten(node):
    """Cognitive complexity adds 1 for recursion; cyclomatic never notices
    that the function calls itself."""
    if not isinstance(node, list):
        return [node]
    out = []
    for child in node:
        out.extend(flatten(child))
    return out
'''

MATCH_STATEMENT = '''
def handle(event):
    """radon charges one per case (a bare `_` is free); the cognitive
    complexity library has no rule for `match`, so it scores zero."""
    match event:
        case {"type": "click", "target": target}:
            return f"click:{target}"
        case {"type": "key", "code": code} if code < 32:
            return "control-key"
        case {"type": "key"}:
            return "key"
        case [first, *rest]:
            return f"batch:{len(rest) + 1}"
        case _:
            return "unknown"
'''

ASYNC_WORKER = '''
import asyncio


async def drain(queue, handler, retries=3):
    """`async for` and `await` cost exactly what their blocking forms cost."""
    async for job in queue:
        for attempt in range(retries):
            try:
                await handler(job)
                break
            except asyncio.TimeoutError:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
'''

SCRIPT_MODULE = '''
"""No functions at all: both metrics score the module body itself."""
import os
import sys

roots = sys.argv[1:] or [os.getcwd()]
seen = set()

for root in roots:
    if not os.path.isdir(root):
        print(f"skipping {root}")
        continue
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        seen.add(name)

if not seen:
    sys.exit(1)
print(len(seen))
'''

LONG_BUT_FLAT = '''
"""Long and boring: twelve helpers over fifty lines, and the module still
scores near zero. Size and complexity are not the same measurement."""
import math


def to_celsius(f):
    return (f - 32) * 5 / 9


def to_fahrenheit(c):
    return c * 9 / 5 + 32


def to_kelvin(c):
    return c + 273.15


def miles(km):
    return km * 0.621371


def kilometres(mi):
    return mi / 0.621371


def pounds(kg):
    return kg * 2.20462


def kilograms(lb):
    return lb / 2.20462


def hypotenuse(a, b):
    return math.hypot(a, b)


def mean(values):
    return sum(values) / len(values)


def clamp(value, low, high):
    return max(low, min(value, high))


def percent(part, whole):
    return 0.0 if not whole else part / whole * 100


def round_to(value, step):
    return round(value / step) * step
'''

TANGLED_MODULE = '''
"""A module-sized example: constants, a class, helpers, and one routine that
both metrics dislike -- for very different reasons."""
SEVERITIES = ("debug", "info", "warning", "error")
THRESHOLD = 2


class Rule:
    def __init__(self, name, level, enabled=True):
        self.name = name
        self.level = level
        self.enabled = enabled

    def applies_to(self, event):
        if not self.enabled:
            return False
        return event.level >= self.level and self.name in event.tags


def severity_of(level):
    if level < 0 or level >= len(SEVERITIES):
        return "unknown"
    return SEVERITIES[level]


def evaluate(events, rules, quiet=False):
    """Four levels of nesting: cyclomatic counts the branches, cognitive
    multiplies them by how deep they sit."""
    findings = {}
    for event in events:
        if event.level < THRESHOLD:
            continue
        for rule in rules:
            if rule.applies_to(event):
                if event.source not in findings:
                    findings[event.source] = []
                for tag in event.tags:
                    if tag in rule.name:
                        findings[event.source].append((rule.name, tag))
                    elif not quiet:
                        print(f"near miss: {rule.name}/{tag}")
            elif rule.enabled and event.level > THRESHOLD:
                findings.setdefault(event.source, [])
    return findings


def report(findings):
    lines = []
    for source, hits in sorted(findings.items()):
        lines.append(f"{source}: {len(hits)}")
    return "\\n".join(lines) or "clean"


if __name__ == "__main__":
    print(report(evaluate([], [])))
'''

EXAMPLES = {
    "area": STRAIGHT_LINE,
    "grade": FLAT_CHAIN,
    "discount": NESTED_CONDITIONS,
    "is_eligible": BOOLEAN_OPERATORS,
    "sum_of_primes": NESTED_LOOP,
    "cart_class": CLASS_WITH_METHODS,
    "rate_limiter": CLOSURE,
    "summarize": COMPREHENSIONS,
    "load_config": ERROR_HANDLING,
    "flatten": RECURSION,
    "handle_match": MATCH_STATEMENT,
    "async_drain": ASYNC_WORKER,
    "script_module": SCRIPT_MODULE,
    "long_but_flat": LONG_BUT_FLAT,
    "tangled_module": TANGLED_MODULE,
}


def cyclomatic_complexity(code: str) -> int:
    return ComplexityVisitor.from_code(code).total_complexity


def cognitive_complexity(code: str) -> int:
    return get_cognitive_complexity_for_node(ast.parse(code))


def main():
    print(f"{'example':<16}{'cyclomatic':>12}{'cognitive':>12}")
    for name, code in EXAMPLES.items():
        cyclomatic = cyclomatic_complexity(code)
        cognitive = cognitive_complexity(code)
        print(f"{name:<16}{cyclomatic:>12}{cognitive:>12}")


if __name__ == "__main__":
    main()
