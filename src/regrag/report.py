"""Plain markdown output for results/, readable on GitHub without a notebook."""


def fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    columns = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)).replace("|", "\\|") for c in columns) + " |")
    return "\n".join(lines) + "\n"


def key_values(data: dict) -> str:
    return table([{"metric": k, "value": v} for k, v in data.items()], ["metric", "value"])
