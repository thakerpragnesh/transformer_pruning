"""The package's folder layout is an architecture, so it is tested like one.

`pruning_transformer/` is grouped by role, and each subpackage may depend
only on the subpackages listed before it in `LAYERS`. That is what keeps the
grouping meaningful: without a check, a convenient import from `surgery`
into `pipeline` (or from `models` into anything) would quietly turn the
folders back into one tangled namespace, with cycles waiting to happen.
These tests parse the source rather than importing it, so a violation is
reported as the offending import line, not as a circular-import traceback.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "pruning_transformer"

# Lowest layer first. A subpackage may import from any subpackage before it.
LAYERS = ["models", "measurement", "analysis", "selection", "surgery", "pipeline"]


def modules():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def subpackage_of(path: Path):
    parts = path.relative_to(PACKAGE).parts
    return parts[0] if len(parts) > 1 else None


def internal_imports(path: Path):
    """Yield `(lineno, target_subpackage_or_None)` for every import of the
    package from inside `path` -- relative or absolute, `TYPE_CHECKING` or
    not. `None` means the top-level `pruning_transformer` namespace itself."""
    here = path.relative_to(PACKAGE).with_suffix("").parts  # e.g. ("selection", "ffn_selectors")
    package_parts = list(here[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package_parts[: len(package_parts) - (node.level - 1)]
            target = base + (node.module.split(".") if node.module else [])
            yield node.lineno, (target[0] if target else None)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pruning_transformer"):
            rest = node.module.split(".")[1:]
            yield node.lineno, (rest[0] if rest else None)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pruning_transformer"):
                    rest = alias.name.split(".")[1:]
                    yield node.lineno, (rest[0] if rest else None)


def test_every_module_lives_in_a_known_subpackage():
    stray = [
        str(p.relative_to(PACKAGE)) for p in modules()
        if p.name != "__init__.py" and subpackage_of(p) not in LAYERS
    ]
    assert not stray, (
        f"Modules outside the layered subpackages: {stray}. Put new code in one of "
        f"{LAYERS} (or add a layer to LAYERS and the package docstring)."
    )


def test_every_subpackage_is_a_package():
    for layer in LAYERS:
        assert (PACKAGE / layer / "__init__.py").is_file(), f"{layer}/ has no __init__.py"


def test_module_names_are_unique_across_subpackages():
    """Docs and docstrings refer to modules by bare name (`head_analysis`,
    `ffn_surgery`); that only stays unambiguous while names are unique."""
    names = [p.stem for p in modules() if p.name != "__init__.py"]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)


def test_subpackages_only_depend_on_lower_layers():
    violations = []
    for path in modules():
        layer = subpackage_of(path)
        if layer is None:
            continue  # the top-level __init__ re-exports everything, by design
        rank = LAYERS.index(layer)
        for lineno, target in internal_imports(path):
            where = f"{path.relative_to(PACKAGE)}:{lineno}"
            if target is None:
                violations.append(f"{where} imports the top-level package (a cycle)")
            elif target != layer and (target not in LAYERS or LAYERS.index(target) > rank):
                violations.append(f"{where} ({layer}) imports {target}, a higher layer")
    assert not violations, "\n".join(violations)
