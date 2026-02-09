"""
SQL statement splitting helpers for migration execution.

This module provides a semicolon-aware SQL splitter that preserves string
and dollar-quoted function bodies while ignoring comment-only fragments.
"""

from __future__ import annotations

from collections.abc import Iterable


def split_sql_statements(sql: str) -> Iterable[str]:
    """
    Yield executable SQL statements from ``sql``.

    Parameters
    ----------
    sql:
        Raw SQL payload that may contain comments, dollar-quoted blocks, and
        semicolon-delimited statements.

    Returns
    -------
    Iterable[str]
        SQL statements with surrounding whitespace trimmed.
    """
    statements: list[str] = []
    buffer: list[str] = []
    in_single = False
    in_dollar = False
    in_line_comment = False
    dollar_tag = ""
    i = 0
    length = len(sql)

    while i < length:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < length else ""

        if not in_line_comment and not in_single and ch == "$":
            j = i + 1
            tag: list[str] = []
            while (j < length and sql[j].isalnum()) or (j < length and sql[j] == "_"):
                tag.append(sql[j])
                j += 1
            if j < length and sql[j] == "$":
                token = "".join(tag)
                if not in_dollar:
                    in_dollar = True
                    dollar_tag = token
                elif token == dollar_tag:
                    in_dollar = False
                    dollar_tag = ""
                buffer.append(sql[i : j + 1])
                i = j + 1
                continue

        if not in_line_comment and not in_dollar and ch == "'":
            if in_single and nxt == "'":
                buffer.append("''")
                i += 2
                continue
            in_single = not in_single

        if not in_single and not in_dollar:
            if not in_line_comment and ch == "-" and nxt == "-":
                in_line_comment = True
                i += 2
                continue
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                    buffer.append(ch)
                i += 1
                continue

        if ch == ";" and not in_single and not in_dollar and not in_line_comment:
            stmt = "".join(buffer).strip()
            if stmt and _has_meaningful_sql(stmt):
                statements.append(stmt)
            buffer.clear()
            i += 1
            continue

        buffer.append(ch)
        i += 1

    tail = "".join(buffer).strip()
    if tail and _has_meaningful_sql(tail):
        statements.append(tail)

    return statements


def _has_meaningful_sql(statement: str) -> bool:
    """
    Return ``True`` when ``statement`` includes executable SQL.

    Parameters
    ----------
    statement:
        Candidate SQL statement fragment.

    Returns
    -------
    bool
        ``True`` for executable SQL, ``False`` for comment-only fragments.
    """
    non_comment_lines: list[str] = []
    for line in statement.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--") or not stripped:
            continue
        non_comment_lines.append(line)

    if not "".join(non_comment_lines).strip():
        return False

    cleaned: list[str] = []
    in_block = False
    i = 0
    length = len(statement)
    while i < length:
        ch = statement[i]
        nxt = statement[i + 1] if i + 1 < length else ""

        if not in_block and ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue

        if in_block and ch == "*" and nxt == "/":
            in_block = False
            i += 2
            continue

        if not in_block:
            cleaned.append(ch)
        i += 1

    return "".join(cleaned).strip() != ""


__all__ = ["split_sql_statements"]
