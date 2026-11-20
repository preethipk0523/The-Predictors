"""Shared pytest settings.

- Tests whose description starts with "Live test" call the real OpenAI API.
  They are skipped by default; run them with:  RUN_LIVE_TESTS=1 python -m pytest
- Any other test that hits an OpenAI error (invalid key, no internet) is
  reported as SKIPPED instead of FAILED.
"""

import os

import pytest

try:
    import openai

    _OPENAI_ERRORS = (
        openai.AuthenticationError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.PermissionDeniedError,
    )
except ImportError:  # openai not installed
    _OPENAI_ERRORS = ()

RUN_LIVE = os.getenv("RUN_LIVE_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if RUN_LIVE:
        return
    skip_live = pytest.mark.skip(reason="live test: calls the real OpenAI API (set RUN_LIVE_TESTS=1 to run)")
    for item in items:
        obj = getattr(item, "obj", None)
        doc = (obj.__doc__ or "") if obj is not None else ""
        if "live test" in doc.lower():
            item.add_marker(skip_live)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _OPENAI_ERRORS and call.excinfo is not None and call.excinfo.errisinstance(_OPENAI_ERRORS):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] or 0,
            "Skipped: needs a valid OpenAI API key and internet access",
        )