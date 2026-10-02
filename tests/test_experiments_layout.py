"""`experiments/` is split by what a script costs to run, and that split is
only useful while it stays true:

    experiments/
    ├── _mrpc.py        shared harness (both groups use it)
    ├── diagnostics/    no training: seconds to minutes
    └── accuracy/       fine-tune -> prune -> heal on MRPC: needs a GPU

The scripts need HuggingFace and downloads, so this suite never runs them.
It parses them instead, which is enough to check the layout, that every
script is valid Python, and that the `_mrpc` import still resolves from
each script's folder.
"""
import ast
import re
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
GROUPS = ("diagnostics", "accuracy")
STAGE = re.compile(r"^(\d{2})_\w+\.py$")


def scripts():
    return sorted(p for p in EXPERIMENTS.rglob("*.py") if STAGE.match(p.name))


def calls(tree, name):
    """Whether `tree` calls `name(...)` or `x.name(...)` anywhere."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == name) or \
               (isinstance(func, ast.Attribute) and func.attr == name):
                return True
    return False


def imports_mrpc(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "_mrpc" for n in ast.walk(tree))


def test_every_stage_script_lives_in_a_group():
    stray = [str(p.relative_to(EXPERIMENTS)) for p in scripts() if p.parent.name not in GROUPS]
    assert not stray, f"Stage scripts outside {GROUPS}: {stray}"
    top_level = sorted(p.name for p in EXPERIMENTS.glob("*.py"))
    assert top_level == ["_mrpc.py"], f"Only the shared harness belongs at the top: {top_level}"


def test_stage_numbers_are_unique():
    numbers = [STAGE.match(p.name).group(1) for p in scripts()]
    assert len(numbers) == len(set(numbers)), sorted(n for n in numbers if numbers.count(n) > 1)


def test_every_script_is_valid_python():
    for path in scripts():
        ast.parse(path.read_text(), filename=str(path))  # raises SyntaxError with file:line


def test_accuracy_holds_exactly_the_scripts_that_train():
    """`accuracy/` means "fine-tunes a baseline": a script that calls
    `train_baseline` belongs there, and nothing else does."""
    misplaced = []
    for path in scripts():
        trains = calls(ast.parse(path.read_text()), "train_baseline")
        if trains != (path.parent.name == "accuracy"):
            misplaced.append(f"{path.relative_to(EXPERIMENTS)} (trains={trains})")
    assert not misplaced, misplaced


def test_mrpc_importers_point_their_path_at_experiments():
    """`_mrpc.py` is one level above every script. A script moved without
    updating its `sys.path` line would only fail on Colab, so catch it here."""
    broken = []
    for path in scripts():
        source = path.read_text()
        if imports_mrpc(ast.parse(source)) and "Path(__file__).resolve().parents[1]" not in source:
            broken.append(str(path.relative_to(EXPERIMENTS)))
    assert not broken, f"These import _mrpc but don't add experiments/ to sys.path: {broken}"
