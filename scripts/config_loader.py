"""
config_loader.py

Loads and normalizes config/symbols.yaml -- the single source of truth for
what this pipeline polls. Keeps YAML-handling logic out of run_fetch.py and
nse_fetch.py; they call the helpers here instead of hardcoding symbol lists,
lot sizes, and display names.

Path resolution: the yaml is resolved relative to the REPO ROOT (the parent
of this scripts/ dir), the same anchor vault_io.py uses for VAULT_DIR -- so
this works from any clone location and never depends on the current working
directory.

workflow_dispatch override (no config edit needed): set the NSE_SYMBOLS env
var to a comma-separated subset (e.g. "NIFTY,BANKNIFTY") to poll only those
symbols this run. Unset/empty -> all symbols in the config. Unknown names
error out rather than silently polling nothing. The workflow exposes a
`symbols` input that exports NSE_SYMBOLS, so a manual run can narrow the set
from the Actions tab without touching symbols.yaml.
"""

import os

import yaml

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "symbols.yaml",
)

_cache = None  # parsed+normalized config, loaded once per run


def _normalize_key(s: str) -> str:
    return str(s).strip().upper()


def _load_raw() -> dict:
    """Reads and normalizes the yaml once, then caches it."""
    global _cache
    if _cache is not None:
        return _cache

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    symbols = {}
    for name, cfg in (raw.get("symbols") or {}).items():
        key = _normalize_key(name)
        cfg = cfg or {}
        display = (cfg.get("index_display_name") or "").strip()
        symbols[key] = {
            "index_display_name": display,
            "lot_size_fallback": cfg.get("lot_size_fallback"),
        }

    _cache = {
        "symbols": symbols,
        "india_vix_display_name": (raw.get("india_vix_display_name") or "").strip(),
    }
    return _cache


def _selected_from_env(all_names: list) -> list:
    """Applies the NSE_SYMBOLS override, preserving config order."""
    override = os.environ.get("NSE_SYMBOLS", "").strip()
    if not override:
        return all_names

    requested = {_normalize_key(s) for s in override.split(",") if s.strip()}
    if not requested:
        return all_names

    unknown = requested - set(all_names)
    if unknown:
        raise ValueError(
            f"NSE_SYMBOLS names not present in symbols.yaml: {sorted(unknown)}. "
            f"Known symbols: {all_names}"
        )
    return [name for name in all_names if name in requested]


# ---------------------------------------------------------------------------
# Public helpers -- callers use these instead of hardcoding.
# ---------------------------------------------------------------------------

def symbol_names() -> list:
    """Symbols to poll this run: all configured, narrowed by NSE_SYMBOLS if set."""
    all_names = list(_load_raw()["symbols"].keys())
    return _selected_from_env(all_names)


def all_symbol_names() -> list:
    """Every symbol in the config, ignoring any NSE_SYMBOLS override."""
    return list(_load_raw()["symbols"].keys())


def lot_size(symbol: str):
    entry = _load_raw()["symbols"].get(_normalize_key(symbol), {})
    return entry.get("lot_size_fallback")


def index_display_name(symbol: str):
    entry = _load_raw()["symbols"].get(_normalize_key(symbol), {})
    return entry.get("index_display_name")


def india_vix_display_name() -> str:
    return _load_raw()["india_vix_display_name"]
