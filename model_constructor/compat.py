from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .errors import CompatibilityError


MIN_TORCH_VERSION: Final[tuple[int, int]] = (2, 2)


def _parse_major_minor(version: str) -> tuple[int, int]:
    core = version.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid version string: {version!r}")
    return int(parts[0]), int(parts[1])


def check_compatibility() -> None:
    try:
        major, minor = _parse_major_minor(torch.__version__)
    except Exception as exc:  # pragma: no cover
        raise CompatibilityError(f"Unable to parse torch version: {torch.__version__!r}") from exc

    if (major, minor) < MIN_TORCH_VERSION:
        raise CompatibilityError(
            f"Unsupported torch version {torch.__version__!r}; requires torch>={MIN_TORCH_VERSION[0]}.{MIN_TORCH_VERSION[1]}"
        )

