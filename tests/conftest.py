"""Shared test setup: puts the repository root (for `app`), src/, scripts/ and tests/ on the import path."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
