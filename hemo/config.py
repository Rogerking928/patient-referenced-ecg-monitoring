"""Configuration loading. One YAML file drives the whole pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        v = dict.get(self, k, default)
        return Cfg(v) if isinstance(v, dict) else v


def _deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def load(path: str | os.PathLike | None = None, overrides=None) -> Cfg:
    """Load the base configuration, then apply an override file if one is given.

    The override may also come from the HEMO_CFG environment variable, so every
    script in the project can be re-run against a different world without editing
    it.  Overrides are deep-merged, so an override file only has to name the keys
    it changes.
    """
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    ov = overrides if overrides is not None else os.environ.get("HEMO_CFG")
    if ov:
        ov_path = Path(ov)
        if not ov_path.is_absolute():
            ov_path = ROOT / ov_path
        with open(ov_path) as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh))
        raw["_override"] = str(ov_path)
    cfg = Cfg(raw)
    cfg["_path"] = str(path)
    for key, sub in (("out_dir", "run"), ("fig_dir", "run")):
        cfg["run"][key] = str(ROOT / cfg["run"][key])
    Path(cfg["run"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["run"]["fig_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def out(cfg: Cfg, *parts) -> Path:
    p = Path(cfg.run.out_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fig(cfg: Cfg, *parts) -> Path:
    p = Path(cfg.run.fig_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
