"""
Tests for the freeze and the locked-test guards (src/freeze_protocol.py, src/run_locked_test.py).

These check that the locked test cannot be opened by accident: it needs a freeze, stops if a
frozen file or a library changed, and cannot run twice. They never read the real locked-test
file and never run the locked test.
"""
import ast
import json
import os

import pytest

import freeze_protocol
import run_locked_test
from provenance import library_versions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")


def _local_imports(path):
    """Names of the modules in src/ that this file imports."""
    tree = ast.parse(open(path, encoding="utf8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if os.path.exists(os.path.join(SRC, n + ".py"))}


def _needed_modules(entry_points):
    seen, todo = set(), list(entry_points)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo.extend(_local_imports(os.path.join(SRC, name + ".py")))
    return seen


# ---------------------------------------------------------------------------
# What is frozen
# ---------------------------------------------------------------------------
def test_every_source_file_the_locked_test_depends_on_is_in_the_frozen_list():
    needed = _needed_modules(["run_locked_test", "run_stage2", "run_stage1", "freeze_protocol"])
    frozen = {os.path.basename(f)[:-3] for f in freeze_protocol.FROZEN_FILES
              if f.startswith("src/") and f.endswith(".py")}
    assert needed <= frozen, f"not frozen although the locked test imports them: {sorted(needed - frozen)}"


def test_every_frozen_file_exists_in_the_repository():
    missing = [f for f in freeze_protocol.FROZEN_FILES if not os.path.exists(os.path.join(ROOT, f))]
    assert not missing, f"listed as frozen but missing: {missing}"


def test_the_frozen_list_has_no_duplicates():
    assert len(freeze_protocol.FROZEN_FILES) == len(set(freeze_protocol.FROZEN_FILES))


# ---------------------------------------------------------------------------
# The development script never touches the locked test
# ---------------------------------------------------------------------------
def _files_read_by(path):
    """Source text of the first argument of every read_csv / joblib.load / open call in a script."""
    tree = ast.parse(open(path, encoding="utf8").read())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("read_csv", "load", "open"):
                found.append(ast.unparse(node.args[0]))
    return found


def test_development_scripts_never_read_the_locked_test_file():
    # run_stage1.py writes test.csv when it makes the split but never reads or scores it.
    for script in ("run_stage1.py", "run_stage2.py", "audit.py"):
        reads = _files_read_by(os.path.join(SRC, script))
        assert not [r for r in reads if "test.csv" in r], script
    with open(os.path.join(SRC, "run_stage2.py"), encoding="utf8") as f:
        assert "test.csv" not in f.read()


def test_the_locked_test_script_checks_the_freeze_before_anything_else():
    tree = ast.parse(open(os.path.join(SRC, "run_locked_test.py"), encoding="utf8").read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    statements = [n for n in main.body if not _is_docstring(n)]
    first = statements[0]
    assert isinstance(first, ast.Assign) and isinstance(first.value, ast.Call)
    assert first.value.func.id == "ensure_ready"


def _is_docstring(node) -> bool:
    """True for a bare string expression, which is how a docstring appears in the syntax tree."""
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------
def _write_freeze(root, files):
    record = {"frozen_at_utc": "2026-01-01 00:00:00", "run_id": "run-x",
              "versions": library_versions(),
              "files": {rel: {"sha256": freeze_protocol.sha256_of(os.path.join(root, rel)),
                              "bytes": 1} for rel in files}}
    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, freeze_protocol.FREEZE_PATH), "w") as f:
        json.dump(record, f)


def test_locked_test_needs_a_freeze_and_stops_if_a_frozen_file_changes(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    with pytest.raises(SystemExit, match="not frozen"):
        run_locked_test.ensure_ready(root)
    _write_freeze(root, ["protocol.md"])
    assert run_locked_test.ensure_ready(root)["run_id"] == "run-x"   # intact: allowed
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan, quietly edited")
    with pytest.raises(SystemExit, match="changed since the freeze"):
        run_locked_test.ensure_ready(root)


def test_locked_test_stops_if_a_frozen_file_is_deleted(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    _write_freeze(root, ["protocol.md"])
    os.remove(tmp_path / "protocol.md")
    with pytest.raises(SystemExit, match="missing: protocol.md"):
        run_locked_test.ensure_ready(root)


def test_locked_test_stops_if_the_libraries_differ_from_the_freeze(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    _write_freeze(root, ["protocol.md"])
    run_locked_test.ensure_ready(root)                      # same libraries: allowed
    path = tmp_path / freeze_protocol.FREEZE_PATH
    record = json.loads(path.read_text())
    record["versions"]["scikit-learn"] = "0.0.1"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="libraries differ") as stop:
        run_locked_test.ensure_ready(root)
    assert "scikit-learn" in str(stop.value)
    record["versions"].pop("scikit-learn")                  # a freeze record with no version is refused too
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="libraries differ"):
        run_locked_test.ensure_ready(root)


def test_locked_test_cannot_run_twice(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    _write_freeze(root, ["protocol.md"])
    with open(tmp_path / freeze_protocol.LOCKED_RESULTS_PATH, "w") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="already been run"):
        run_locked_test.ensure_ready(root)


def test_nothing_can_be_frozen_after_the_test_has_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("reports")
    with open(freeze_protocol.LOCKED_RESULTS_PATH, "w") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="already been run"):
        freeze_protocol.main()


def test_an_existing_freeze_cannot_be_overwritten_by_running_freeze_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("reports")
    with open(freeze_protocol.FREEZE_PATH, "w") as f:
        json.dump({"frozen_at_utc": "2026-01-01 00:00:00"}, f)
    with pytest.raises(SystemExit, match="already frozen"):
        freeze_protocol.main()
