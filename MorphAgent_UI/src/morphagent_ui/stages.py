"""Qt-independent parsing shared by the legacy UI and local workspace."""
import re


class StageDetector:
    """Translate the existing CLI's stable top-level messages into UI stages."""

    _rules = (
        (re.compile(r"^Step 2\.[45]:"), 1),
        (re.compile(r"^Step 3(?:\.5)?:"), 2),
        (re.compile(r"^Step 4:"), 3),
        (re.compile(r"^Step (?:5(?:\.5)?|6):"), 4),
        (re.compile(r"^\[DONE\] All \d+ rounds .*complete!"), 5),
        (re.compile(r"^Final feature file:"), 5),
        (re.compile(r"^Step [12]:"), 0),
    )

    def __init__(self) -> None:
        self.index = 0

    def feed(self, raw_line: str) -> int | None:
        line = raw_line.rstrip()
        # Nested code-route messages intentionally begin with spaces and reuse
        # "Step 2/3/4"; only unindented top-level stages are eligible.
        if not line or line[0].isspace():
            return None
        for pattern, index in self._rules:
            if pattern.search(line):
                if index < self.index:
                    return None
                changed = index != self.index
                self.index = index
                return index if changed or index == 0 else None
        return None
