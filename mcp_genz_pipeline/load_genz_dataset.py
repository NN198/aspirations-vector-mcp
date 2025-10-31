"""Utilities for loading a locally-downloaded Gen-Z dataset into the MCP server.

The MCP server exposes a ``load_genz`` tool that should call :func:`load_genz_dataset`
to reuse this implementation. This updated version assumes that the raw dataset has
already been downloaded manually (for example from Kaggle's website) and placed on
disk. It therefore focuses on discovering that file, emitting structured diagnostics
for connected MCP clients, and normalising the contents into the shared
``data/genz_career_aspirations.csv`` location for downstream tools.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


OUTPUT_PATH = Path("data/genz_career_aspirations.csv")
ENV_SOURCE_PATH = "GENZ_DATASET_SOURCE"
SOURCE_CANDIDATES: Sequence[Path] = (
    Path("data/understanding-career-aspirations-of-genz.csv"),
    Path("data/understanding-career-aspirations-of-genz/understanding-career-aspirations-of-genz.csv"),
    Path("data/understanding-career-aspirations-of-genz/Understanding Career Aspirations of Gen-Z.csv"),
    Path("data/raw/understanding-career-aspirations-of-genz.csv"),
    Path("data/raw/understanding-career-aspirations-of-genz/understanding-career-aspirations-of-genz.csv"),
    Path("data/raw/understanding-career-aspirations-of-genz/Understanding Career Aspirations of Gen-Z.csv"),
)


def _iter_candidate_paths() -> Iterable[Path]:
    env_path = os.getenv(ENV_SOURCE_PATH)
    if env_path:
        yield Path(env_path).expanduser()
    yield from SOURCE_CANDIDATES
    data_dir = Path("data")
    if data_dir.exists():
        yield from sorted(data_dir.rglob("*.csv"))


def _resolve_source_path(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(
            f"Explicit dataset path {explicit_path!s} does not exist."
        )

    for candidate in _iter_candidate_paths():
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Unable to locate the Gen-Z dataset CSV. Download the dataset manually and "
        "either place the main CSV within the repository's data/ directory or set the "
        f"{ENV_SOURCE_PATH} environment variable to the desired file."
    )


def _load_with_pandas(csv_path: Path, target_path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return False

    df = pd.read_csv(csv_path)
    df = df.fillna(value=pd.NA).convert_dtypes()

    preview = df.head()
    print(json.dumps({"shape": df.shape, "columns": df.columns.tolist()}, indent=2))
    print(preview.to_string(index=False))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return True


def _format_preview(rows: List[dict], columns: Sequence[str]) -> str:
    if not rows:
        return "(dataset is empty)"

    def _cell_text(row: dict, column: str) -> str:
        value = row.get(column, "")
        if value is None:
            return ""
        return str(value)

    col_widths = {
        column: max(len(column), *(len(_cell_text(row, column)) for row in rows))
        for column in columns
    }

    def _format_row(row: dict) -> str:
        return " ".join(_cell_text(row, column).ljust(col_widths[column]) for column in columns)

    header = " ".join(column.ljust(col_widths[column]) for column in columns)
    divider = " ".join("-" * col_widths[column] for column in columns)
    body = "\n".join(_format_row(row) for row in rows)
    return f"{header}\n{divider}\n{body}"


def _load_without_pandas(csv_path: Path, target_path: Path) -> Path:
    preview_rows: List[dict] = []
    columns: List[str] = []
    row_count = 0

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as source_file, target_path.open(
        "w", encoding="utf-8", newline=""
    ) as target_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError(
                f"The CSV file {csv_path!s} is missing a header row and cannot be processed."
            )

        columns = list(reader.fieldnames)
        writer = csv.DictWriter(target_file, fieldnames=columns)
        writer.writeheader()

        for row in reader:
            row_count += 1
            cleaned = {key: (value if value != "" else None) for key, value in row.items()}
            if len(preview_rows) < 5:
                preview_rows.append(cleaned)
            writer.writerow({key: "" if cleaned[key] is None else cleaned[key] for key in columns})

    print(json.dumps({"shape": (row_count, len(columns)), "columns": columns}, indent=2))
    print(_format_preview(preview_rows, columns))

    return target_path


def load_genz_dataset(
    source_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Normalise a local copy of the Gen-Z dataset for the MCP ``load_genz`` tool.

    Parameters
    ----------
    source_path:
        Optional override for the source CSV file. When omitted the function looks
        for a dataset path in the :data:`GENZ_DATASET_SOURCE` environment variable and
        a small set of common locations inside the repository's ``data/`` directory.
    save_path:
        Optional override for the destination CSV path. Defaults to
        ``data/genz_career_aspirations.csv``.

    Returns
    -------
    Path
        Path to the saved dataset CSV suitable for downstream MCP tooling.
    """

    resolved_source = _resolve_source_path(source_path)
    target_path = Path(save_path) if save_path is not None else OUTPUT_PATH

    if _load_with_pandas(resolved_source, target_path):
        return target_path

    # Pandas is unavailable; fall back to a streaming CSV implementation.
    return _load_without_pandas(resolved_source, target_path)


if __name__ == "__main__":
    destination = load_genz_dataset()
    print(f"Dataset saved to {destination.resolve()}")
