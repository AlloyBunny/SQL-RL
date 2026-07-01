"""BIRD-compatible SQL repair prompt and response parsing."""

from __future__ import annotations

import re
from typing import Any

from sql_rl.data.six_gym import as_list


SYSTEM_PROMPT = """You are an expert SQL debugger. Your task is to identify and fix bugs in SQL queries.

When given a problematic SQL query, you should:
1. Analyze the query to identify what's wrong
2. Provide detailed, high-level reasoning about the problem and solution
3. Generate a corrected SQL query

Structure your response using this format:

<thought>
Provide detailed, high-level reasoning about the problem(s) and how to fix it.
</thought>

<solution>
[Provide the corrected SQL code here, without markdown code fences]
</solution>

Remember:
- Be specific about what the error is and why it occurs
- Provide detailed reasoning in your thought process
- Ensure the solution SQL is complete and executable
- Do NOT include markdown code fences (```sql) in the solution tag"""


USER_PROMPT = """## Problem Description
{query}

## Database Schema
{schema}

## Problematic SQL
The following SQL code is problematic and does not satisfy the user's requirements. It may contain syntax errors, logic errors, or produce incorrect results:

```sql
{issue_sql}
```

Please analyze the problematic SQL and provide your solution."""


_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.IGNORECASE | re.DOTALL)


def build_user_prompt(query: str, schema: str, issue_sql: str | list[str]) -> str:
    return USER_PROMPT.format(
        query=query.strip(),
        schema=schema.strip(),
        issue_sql="\n\n".join(as_list(issue_sql)).strip(),
    )


def build_chat_messages(record: dict[str, Any], schema: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                query=str(record.get("query", "")),
                schema=schema,
                issue_sql=as_list(record.get("issue_sql")),
            ),
        },
    ]


def extract_solution_sql(response: str) -> str:
    """Extract final SQL from a model response.

    The BIRD format uses `<solution>...</solution>`. For debugging and baseline
    scripts, this also accepts a raw SQL response.
    """
    if not response:
        return ""
    match = _SOLUTION_RE.search(response)
    sql = match.group(1) if match else response
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()

