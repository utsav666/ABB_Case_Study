from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MATPLOTLIB_CONFIG_DIR = "/tmp/matplotlib"


def ensure_matplotlib_runtime_config() -> None:
    """Set a writable Matplotlib config directory for containerized runtimes."""
    configured_directory = os.getenv(
        "MPLCONFIGDIR",
        DEFAULT_MATPLOTLIB_CONFIG_DIR,
    ).strip()
    if not configured_directory:
        configured_directory = DEFAULT_MATPLOTLIB_CONFIG_DIR

    os.environ["MPLCONFIGDIR"] = configured_directory
    Path(configured_directory).mkdir(parents=True, exist_ok=True)
