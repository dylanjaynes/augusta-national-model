"""
Event tier weighting for rolling SG features.
Higher-tier events count more in the exponential decay window.
"""

# (pattern_substring_lowercase, weight)
_TIER_RULES = [
    # Tier 1 — Masters (3.0)
    ("masters", 3.0),
    # Tier 3 — Other majors (1.8) — checked before Tier 2 patterns that might overlap
    ("u.s. open", 1.8), ("us open", 1.8),
    ("the open championship", 1.8), ("open championship", 1.8),
    ("pga championship", 1.8),
    # Tier 2 — Strongest non-major fields (2.0)
    ("players championship", 2.0),
    ("wgc", 2.0), ("world golf championships", 2.0),
    ("tour championship", 2.0), ("bmw championship", 2.0),
    ("fedex", 2.0),
    # Tier 4 — Elevated / signature events (1.4)
    ("genesis", 1.4), ("arnold palmer", 1.4), ("memorial tournament", 1.4),
    ("rbc heritage", 1.4), ("travelers", 1.4), ("sentry", 1.4),
    ("at&t pebble beach", 1.4), ("valspar", 1.4), ("bay hill", 1.4),
    ("wells fargo", 1.4), ("charles schwab", 1.4),
    ("rocket mortgage", 1.4), ("john deere", 1.4), ("3m open", 1.4),
    ("scottish open", 1.4), ("irish open", 1.4),
    ("zozo", 1.4), ("hero world challenge", 1.4),
    ("american express", 1.4), ("cognizant", 1.4),
    # Tier 6 — Opposite-field / weak-field (0.5)
    ("barracuda", 0.5), ("barbasol", 0.5), ("puerto rico", 0.5),
    ("corales", 0.5), ("puntacana", 0.5), ("bermuda", 0.5),
    ("butterfield", 0.5), ("sanderson farms", 0.5), ("fortinet", 0.5),
    ("shriners", 0.5), ("rsm classic", 0.5),
    ("korn ferry", 0.5), ("nationwide", 0.5),
    # Tier 7 — LIV Golf (0.4)
    ("liv", 0.4), ("invitational series", 0.4),
]


def get_event_weight(event_name: str) -> float:
    """Return tier weight for an event. Higher = more important field.
    Uses case-insensitive substring matching. If multiple match,
    returns the highest weight."""
    if not event_name or not isinstance(event_name, str):
        return 1.0
    lower = event_name.lower()
    best = 1.0  # default = Tier 5 (standard)
    for pattern, weight in _TIER_RULES:
        if pattern in lower:
            best = max(best, weight)
    return best


def classify_events(event_names):
    """Classify a list of event names and return a dict of {name: weight}."""
    return {name: get_event_weight(name) for name in event_names}
