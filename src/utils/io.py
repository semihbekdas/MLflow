import json
from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: Dict[str, Any], override_json: str) -> Dict[str, Any]:
    """Shallow/partial deep merge config with keys provided as a JSON string.
    Example override_json: '{"model.params.n_estimators": 200}'
    Supports dotted keys for 1-level nested dicts.
    """
    if not override_json:
        return cfg
    try:
        overrides = json.loads(override_json)
    except json.JSONDecodeError:
        raise ValueError("--override must be a valid JSON string")

    for k, v in overrides.items():
        if "." in k:
            parts = k.split(".")
            ref = cfg
            for p in parts[:-1]:
                if p not in ref or not isinstance(ref[p], dict):
                    ref[p] = {}
                ref = ref[p]
            ref[parts[-1]] = v
        else:
            cfg[k] = v
    return cfg


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
