import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools.check_clean_code import (
    COMMENT_REASON,
    DOCSTRING_REASON,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    Issue,
    check_source,
    format_issue,
    run,
)

LIMIT = 3
QUOTE = '"' * 3


def join_lines(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def function_with_body_lines(body_lines: int) -> str:
    return join_lines("def sample() -> None:", *["    pass"] * body_lines)


class CheckSourceTests(unittest.TestCase):
    def test_clean_source_has_no_issues(self) -> None:
        source = join_lines("def add(a: int, b: int) -> int:", "    return a + b")
        self.assertEqual(check_source(source, LIMIT), [])

    def test_detects_comment_with_its_line(self) -> None:
        source = join_lines("x = 1", "y = 2  " + chr(35) + " nota")
        self.assertEqual(check_source(source, LIMIT), [Issue(2, COMMENT_REASON)])

    def test_hash_inside_string_is_not_a_comment(self) -> None:
        source = join_lines("x = '" + chr(35) + " no es comentario'")
        self.assertEqual(check_source(source, LIMIT), [])

    def test_detects_module_class_and_function_docstrings(self) -> None:
        source = join_lines(
            f"{QUOTE}modulo{QUOTE}",
            "class Sample:",
            f"    {QUOTE}clase{QUOTE}",
            "    def method(self) -> None:",
            f"        {QUOTE}metodo{QUOTE}",
        )
        lines = [issue.line for issue in check_source(source, LIMIT) if issue.reason == DOCSTRING_REASON]
        self.assertEqual(lines, [1, 3, 5])

    def test_string_after_first_statement_is_not_a_docstring(self) -> None:
        source = join_lines("def sample() -> None:", "    x = 1", "    'texto'")
        self.assertEqual(check_source(source, LIMIT), [])

    def test_function_at_limit_is_accepted(self) -> None:
        self.assertEqual(check_source(function_with_body_lines(LIMIT), LIMIT), [])

    def test_function_over_limit_is_reported(self) -> None:
        issues = check_source(function_with_body_lines(LIMIT + 1), LIMIT)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].line, 1)
        self.assertIn("'sample'", issues[0].reason)

    def test_format_issue_uses_path_line_reason(self) -> None:
        self.assertEqual(format_issue("a.py", Issue(7, "motivo")), "a.py:7: motivo")


class RunTests(unittest.TestCase):
    def run_on_source(self, source: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text(source, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run([str(path)])
        return code, output.getvalue()

    def test_clean_file_exits_with_success_and_prints_nothing(self) -> None:
        code, output = self.run_on_source(join_lines("x = 1"))
        self.assertEqual(code, EXIT_SUCCESS)
        self.assertEqual(output, "")

    def test_dirty_file_exits_with_failure_and_prints_location(self) -> None:
        code, output = self.run_on_source(join_lines(chr(35) + " nota"))
        self.assertEqual(code, EXIT_FAILURE)
        self.assertIn("sample.py:1: " + COMMENT_REASON, output)


if __name__ == "__main__":
    unittest.main()
