import argparse
import ast
import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

MAX_FUNCTION_LINES = 25
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
COMMENT_REASON = "comentario encontrado"
DOCSTRING_REASON = "docstring encontrado"
LONG_FUNCTION_REASON = "la función '{name}' tiene {length} líneas de cuerpo (máximo {limit})"
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, *FUNCTION_NODES)


@dataclass(frozen=True)
class Issue:
    line: int
    reason: str


def comment_issues(source: str) -> list[Issue]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [Issue(token.start[0], COMMENT_REASON) for token in tokens if token.type == tokenize.COMMENT]


def is_docstring(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Constant):
        return False
    return isinstance(statement.value.value, str)


def docstring_issues(tree: ast.Module) -> list[Issue]:
    owners = [node for node in ast.walk(tree) if isinstance(node, DOCSTRING_OWNERS)]
    return [
        Issue(owner.body[0].lineno, DOCSTRING_REASON)
        for owner in owners
        if owner.body and is_docstring(owner.body[0])
    ]


def function_body_length(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return function.end_lineno - function.body[0].lineno + 1


def long_function_issues(tree: ast.Module, limit: int) -> list[Issue]:
    functions = [node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)]
    return [
        Issue(function.lineno, LONG_FUNCTION_REASON.format(name=function.name, length=length, limit=limit))
        for function in functions
        if (length := function_body_length(function)) > limit
    ]


def check_source(source: str, limit: int) -> list[Issue]:
    tree = ast.parse(source)
    issues = comment_issues(source) + docstring_issues(tree) + long_function_issues(tree, limit)
    return sorted(issues, key=lambda issue: issue.line)


def format_issue(path: str, issue: Issue) -> str:
    return f"{path}:{issue.line}: {issue.reason}"


def exit_code_for(issue_count: int) -> int:
    return EXIT_FAILURE if issue_count > 0 else EXIT_SUCCESS


def read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def print_issues(path: str, issues: list[Issue]) -> None:
    for issue in issues:
        print(format_issue(path, issue))


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica las reglas de código limpio del proyecto.")
    parser.add_argument("paths", nargs="+", help="archivos .py a revisar")
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    issue_count = 0
    for path in arguments.paths:
        issues = check_source(read_source(path), MAX_FUNCTION_LINES)
        print_issues(path, issues)
        issue_count += len(issues)
    return exit_code_for(issue_count)


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
