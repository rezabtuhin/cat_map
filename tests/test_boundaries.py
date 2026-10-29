import ast
from pathlib import Path


def test_domain_does_not_import_frameworks_or_storage():
    root = Path(__file__).resolve().parents[1] / "src/category_matcher"
    forbidden = {
        "sqlite3",
        "fastapi",
        "torch",
        "sentence_transformers",
        "category_matcher.runtime",
        "category_matcher.api",
        "category_matcher.storage",
        "category_matcher.matcher",
    }
    for node in ast.walk(ast.parse((root / "domain.py").read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        assert all(not any(n == f or n.startswith(f + ".") for f in forbidden) for n in names)


def test_api_does_not_execute_sql():
    root = Path(__file__).resolve().parents[1] / "src/category_matcher/api"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"execute", "executemany", "executescript"}, str(path)
