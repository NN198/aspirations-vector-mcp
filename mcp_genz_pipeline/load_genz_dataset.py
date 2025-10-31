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