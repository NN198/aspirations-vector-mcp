
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

"""Utility for retrieving the Gen-Z Career Aspirations dataset for the MCP server.

This module is designed to be called from the MCP server's ``load_genz`` tool so
that the server can reuse the shared implementation for downloading, cleaning,
inspecting, and persisting the Kaggle dataset locally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import pandas as pd 
except ImportError as exc:  # pragma: no cover - handled at runtime when dependency missing
    raise SystemExit("pandas must be installed before running this script.") from exc

try:
    from kaggle.api.kaggle_api_extended import KaggleApi #type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime when dependency missing
    raise SystemExit("The 'kaggle' package must be installed before running this script.") from exc
    import kagglehub
    from kagglehub.adapters import KaggleDatasetAdapter
except ImportError as exc:  # pragma: no cover - handled at runtime when dependency missing
    raise SystemExit(
        "kagglehub with the pandas extra must be installed before running this script."
    ) from exc



DATASET_SLUG = "kulturehire/understanding-career-aspirations-of-genz"
OUTPUT_PATH = Path("data/genz_career_aspirations.csv")


def _detect_main_csv(dataset_dir: Path) -> Path:
    """Return the most likely CSV file containing the main dataset.

    The helper inspects all CSV files in the downloaded directory and selects the
    largest one, assuming that it represents the main table. A :class:`FileNotFoundError`
    is raised if no CSV files are detected.
    """

    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found after downloading dataset to {dataset_dir!s}."
        )
    return max(csv_files, key=lambda csv_file: csv_file.stat().st_size)



def _resolve_latest_version(adapter: KaggleDatasetAdapter) -> str:
    """Determine the identifier for the latest dataset version."""

    if hasattr(adapter, "get_latest_version"):
        return adapter.get_latest_version()  # type: ignore[return-value]
    latest_attr = getattr(adapter, "latest_version", "latest")
    return latest_attr() if callable(latest_attr) else latest_attr  # type: ignore[return-value]


def load_genz_dataset(save_path: Optional[Path] = None) -> Path:
    """Download, clean, and persist the Gen-Z career aspirations dataset.

    Notes
    -----
    The function prints a short summary to STDOUT so that MCP tooling can stream
    progress information back to connected clients.

    Parameters
    ----------
    save_path:
        Optional custom location for the persisted dataset. If omitted the default
        ``data/genz_career_aspirations.csv`` is used. The path will be created if it
        does not already exist.


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

        Path to the saved CSV file, which can be used by downstream MCP tools.
    """

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - requires user configuration
        raise SystemExit(
            "Failed to authenticate with Kaggle. Ensure kaggle.json credentials are configured."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="genz_career_aspirations_") as temp_dir:
        temp_path = Path(temp_dir)
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(temp_path),
            unzip=True,
            quiet=False,
        )

        csv_path = _detect_main_csv(temp_path)

        df = pd.read_csv(csv_path)
        df = df.fillna(value=pd.NA).convert_dtypes()

        print(json.dumps({"shape": df.shape, "columns": df.columns.tolist()}, indent=2))
        print(df.head().to_string(index=False))

        target_path = Path(save_path) if save_path is not None else OUTPUT_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)

    adapter = KaggleDatasetAdapter(DATASET_SLUG)
    print("Using kagglehub version {}".format(getattr(kagglehub, '__version__', 'unknown')))
    latest_version = _resolve_latest_version(adapter)

    if hasattr(adapter, "download"):
        dataset_dir = Path(adapter.download(version=latest_version))
    elif hasattr(adapter, "download_version"):
        dataset_dir = Path(adapter.download_version(version=latest_version))
    else:  # pragma: no cover - defensive fallback for unexpected adapter API changes
        raise RuntimeError("Unable to download dataset with the available KaggleHub adapter.")

    csv_path = _detect_main_csv(dataset_dir)

    df = pd.read_csv(csv_path)
    df = df.fillna(value=pd.NA).convert_dtypes()

    print(json.dumps({"shape": df.shape, "columns": df.columns.tolist()}, indent=2))
    print(df.head().to_string(index=False))

    target_path = Path(save_path) if save_path is not None else OUTPUT_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)


    return target_path


if __name__ == "__main__":
    saved_path = load_genz_dataset()
    print(f"Dataset saved to {saved_path.resolve()}")