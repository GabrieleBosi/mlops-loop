"""Config and environment loading. No dependency on python-dotenv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """The directory that holds configs/ and mlops_loop/."""
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | str | None = None) -> list[str]:
    """Set any variable named in .env that is not already in the environment.

    Values already in the environment win, so a shell export or CI secret is never
    overwritten by a file. Returns the names that were set, never the values.
    """
    dotenv = Path(path) if path is not None else repo_root() / ".env"
    if not dotenv.exists():
        return []
    applied: list[str] = []
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML config. Stops if the file is missing or is not a mapping."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config {p} must be a mapping, got {type(data).__name__}")
    return data


def skeleton_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configs/skeleton.yaml and check the keys the pipeline needs are present."""
    p = Path(path) if path is not None else repo_root() / "configs" / "skeleton.yaml"
    cfg = load_yaml(p)
    required = {
        "data": ["url"],
        "split": ["holdout_fraction", "val_fraction", "seed"],
        "model": ["C", "solver", "max_iter"],
        "registry": ["name", "alias"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"{p}: missing section '{section}'")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise ValueError(f"{p}: section '{section}' is missing {missing}")
    return cfg


def drift_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configs/drift.yaml. Session 1 only reads the future_batch rule."""
    p = Path(path) if path is not None else repo_root() / "configs" / "drift.yaml"
    cfg = load_yaml(p)
    batches = cfg.get("future_batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError(f"{p}: future_batches must be a non-empty list")
    for entry in batches:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("rules"):
            raise ValueError(f"{p}: every future batch needs a name and rules: {entry}")
    psi = cfg.get("psi")
    if not isinstance(psi, dict) or not isinstance(psi.get("thresholds"), dict):
        raise ValueError(f"{p}: psi.thresholds must be a mapping of feature to threshold")
    if not isinstance(psi.get("prediction_threshold"), (int, float)):
        raise ValueError(f"{p}: psi.prediction_threshold must be a number")
    promotion = cfg.get("promotion")
    if not isinstance(promotion, dict) or not isinstance(promotion.get("margin"), (int, float)):
        raise ValueError(f"{p}: promotion.margin must be a number")
    return cfg


def sweep_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configs/sweep.yaml and check the shape the sweep depends on."""
    p = Path(path) if path is not None else repo_root() / "configs" / "sweep.yaml"
    cfg = load_yaml(p)
    selection = cfg.get("selection")
    if not isinstance(selection, dict) or "metric" not in selection or "split" not in selection:
        raise ValueError(f"{p}: selection must set 'metric' and 'split'")
    if not isinstance(cfg.get("configs"), list) or not cfg["configs"]:
        raise ValueError(f"{p}: configs must be a non-empty list")
    return cfg


def thresholds_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configs/thresholds.yaml. An empty or malformed gate is a failure, not a pass."""
    p = Path(path) if path is not None else repo_root() / "configs" / "thresholds.yaml"
    cfg = load_yaml(p)
    metrics = cfg.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"{p}: metrics must be a non-empty mapping, or the gate passes nothing")
    return cfg
