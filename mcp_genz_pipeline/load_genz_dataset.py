"""Load the Gen-Z career aspirations dataset for the MCP ``load_genz`` tool.

This script is designed to be imported by the MCP server, which exposes a
``load_genz`` tool. The tool should call :func:`load_genz_dataset` to ensure that
clients always receive a normalised CSV at ``data/genz_career_aspirations.csv``.
The loader now assumes that the dataset has been manually downloaded to
``data/understanding-career-aspirations-of-genz.csv``.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Optional

DEFAULT_SOURCE_PATH = Path("data/understanding-career-aspirations-of-genz.csv")
DEFAULT_TARGET_PATH = Path("data/genz_career_aspirations.csv")
ENV_SOURCE_PATH = "GENZ_DATASET_SOURCE"


def _resolve_source_path(source_path: Optional[Path] = None) -> Path:
    """Determine where the raw dataset lives on disk."""

    candidate_paths: List[Path] = []

    if source_path is not None:
        candidate_paths.append(Path(source_path).expanduser())

    env_value = os.getenv(ENV_SOURCE_PATH)
    if env_value:
        candidate_paths.append(Path(env_value).expanduser())

    candidate_paths.append(DEFAULT_SOURCE_PATH)

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Unable to locate the Gen-Z dataset CSV. Ensure it is saved to "
        f"{DEFAULT_SOURCE_PATH!s} or set the {ENV_SOURCE_PATH} environment variable."
    )


def _try_load_with_pandas(source_path: Path, target_path: Path) -> Optional[Path]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None

    dataframe = pd.read_csv(source_path)
    dataframe = dataframe.convert_dtypes().fillna(value=pd.NA)

    print(json.dumps({"shape": dataframe.shape, "columns": dataframe.columns.tolist()}, indent=2))
    print("First 5 rows (pandas):")
    print(dataframe.head().to_string(index=False))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(target_path, index=False)
    return target_path


def _normalise_cell(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _load_with_csv_module(source_path: Path, target_path: Path) -> Path:
    preview_rows: List[dict] = []
    row_count = 0

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file, target_path.open(
        "w", encoding="utf-8", newline=""
    ) as target_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError(
                f"The CSV file {source_path!s} is missing a header row and cannot be processed."
            )

        columns = list(reader.fieldnames)
        writer = csv.DictWriter(target_file, fieldnames=columns)
        writer.writeheader()

        for row in reader:
            row_count += 1
            normalised = {key: _normalise_cell(value) for key, value in row.items()}
            if len(preview_rows) < 5:
                preview_rows.append(normalised)

            writer.writerow({key: "" if normalised[key] is None else normalised[key] for key in columns})

    print(json.dumps({"shape": (row_count, len(columns)), "columns": columns}, indent=2))
    print("First 5 rows (csv module):")
    print(json.dumps(preview_rows, indent=2))

    return target_path


def load_genz_dataset(
    source_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Normalise and persist the Gen-Z dataset for MCP clients.

    When ``source_path`` is omitted the loader checks the
    :data:`GENZ_DATASET_SOURCE` environment variable before falling back to the
    default CSV location at :data:`DEFAULT_SOURCE_PATH`. ``save_path`` can be
    overridden for testing but otherwise defaults to
    ``data/genz_career_aspirations.csv``.
    """

    resolved_source = _resolve_source_path(source_path)
    target_path = Path(save_path).expanduser() if save_path is not None else DEFAULT_TARGET_PATH

    pandas_result = _try_load_with_pandas(resolved_source, target_path)
    if pandas_result is not None:
        return pandas_result

    return _load_with_csv_module(resolved_source, target_path)


if __name__ == "__main__":
    destination = load_genz_dataset()
    print(f"Dataset saved to {destination.resolve()}")
