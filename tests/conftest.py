"""Skip the tests that need the Parker-Gibson bundle when it is not present.

`cases/parker_gibson/` is Program on Negotiation material and is not distributed with the
repository (README, "Rights"). Every test that loads it is skipped, not failed, so a public
checkout still runs the rest of the suite.
"""
import inspect
from pathlib import Path

import pytest

CASE_FILE = Path(__file__).resolve().parents[1] / "cases" / "parker_gibson" / "case.yaml"
REASON = "cases/parker_gibson is not distributed (PON material); see README, Rights"


def _needs_case(item) -> bool:
    if set(getattr(item, "fixturenames", ())) & {"bundle", "program"}:
        return True
    try:
        return "parker_gibson" in inspect.getsource(item.function)
    except (OSError, TypeError):
        return False


def pytest_collection_modifyitems(config, items):
    if CASE_FILE.exists():
        return
    marker = pytest.mark.skip(reason=REASON)
    for item in items:
        if _needs_case(item):
            item.add_marker(marker)
