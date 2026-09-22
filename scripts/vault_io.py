"""
vault_io.py

Handles everything that touches disk under vault/, and is the single source of
truth for vault *layout* -- compact.py imports the path helpers below rather
than rebuilding paths of its own.

    vault/raw/<SYMBOL>/JSON-YYYY-MM-DD.json.gz   -- one growing file per
        symbol per day; each fetch cycle appends one more entry to a list
        inside the (re-written) gzip file. Unchanged by Phase 3.

    vault/tables/<SYMBOL>/<YYYY-MM-DD>/MAIN-<YYYY-MM-DD>T<HHMMSS>-<poller>.parquet
        -- ONE SHARD PER FETCH CYCLE (Phase 3). Typed parquet, written once,
        never reopened. Replaces the old append-to-a-growing-CSV pattern:
        every cycle used to rewrite a blob that got bigger all day, and a
        crash mid-append could leave a torn trailing row that poisoned the
        whole day's file.

    vault/tables/<SYMBOL>/MAIN-<YYYY-MM-DD>.parquet
        -- the compacted day, produced by scripts/compact.py merging the
        shards above. Deliberately one directory level UP from the shards, so
        no glob ever sees both, and "the day directory is gone" is an
        unambiguous signal that the day has been compacted.

Three properties the shard name is built to have:

1. Deterministic / idempotent. The path is a pure function of the rows' own
   fetch_ts plus the writing poller's id -- nothing derived from "now" at write
   time. Re-writing the same cycle (or replaying it from the raw archive in a
   later phase) regenerates the identical path and overwrites in place instead
   of accumulating duplicates.
2. Sorts chronologically as plain text. The sort-relevant prefix is
   fixed-width -- "MAIN-" (5) + YYYY-MM-DD (10) + "T" (1) + HHMMSS (6) = 22
   characters, every field zero-padded -- so two names can only differ inside
   the poller suffix if their timestamps are byte-identical. The suffix can
   never perturb the ordering of two different instants.
3. Collision-free across pollers. The shard identity key is
   (symbol, fetch_ts, poller_id), not (symbol, fetch_ts): two pollers landing a
   cycle in the same second must not write the same path and silently clobber
   each other. The poller_id *column* alone cannot prevent that, because the
   loser's rows never reach disk at all.

Note for any later phase that needs to parse a shard filename: parse it
POSITIONALLY (stem[5:15] = date, stem[16:22] = time, stem[23:] = poller), never
by splitting on "-" -- both the date and the default poller id "gh-actions"
contain hyphens. Nothing here parses filenames; compact.py reads fetch_ts_utc
out of parquet contents instead.

Also owns the freshness check that runs before anything gets written -- a
bad/empty NSE response should never silently pollute the archive.
"""

import glob
import gzip
import json
import math
import os
import re
from datetime import date, datetime

import pyarrow as pa
import pyarrow.parquet as pq

import schema

VAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vault"
)
RAW_DIR = os.path.join(VAULT_DIR, "raw")
TABLES_DIR = os.path.join(VAULT_DIR, "tables")

# Which poller produced these rows. Resolved ONCE, here, and re-exported to
# run_fetch.py -- so the value that names the shard file is by construction the
# same value that lands in every row's poller_id column. Two independent
# os.environ reads with the same default would agree today, but a filename
# disagreeing with its own poller_id column is exactly the failure the
# poller-in-the-filename design exists to prevent.
POLLER_ID = os.environ.get("POLLER_ID", schema.DEFAULT_POLLER_ID)

MAIN_COLUMNS = [
    "fetch_ts_utc",
    "fetch_ts_ist",
    "symbol",
    "expiry_date",
    "strike",
    "option_type",
    "underlying_value",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
    # Whole-book aggregate depth. bid_qty/ask_qty above are LEVEL-1 only
    # (buyQuantity1/sellQuantity1); these two are NSE's totals across the
    # entire book, so they are a different liquidity signal, not a rename.
    "total_buy_quantity",
    "total_sell_quantity",
    "ltp",
    "mid_price",
    "open_interest",
    "change_in_oi",
    "total_traded_volume",
    "pchange_vs_prev_close",
    # nse_iv is now the ONLY IV column -- the in-house solved IV
    # (formerly "computed_iv") was dropped: it added little value once
    # NSE's own published IV was confirmed available, and it was also the
    # thing most distorted by the near-expiry cost-of-carry instability
    # that motivated switching the dividend-yield source (see run_fetch.py
    # get_dividend_yield_and_carry()).
    "nse_iv",
    "delta", "gamma", "theta", "vega",
    "vanna", "charm", "vomma",
    "speed", "zomma", "color", "veta",
    "omega", "dual_delta", "dual_gamma",
    "time_to_expiry_years",
    "futures_price",
    "implied_cost_of_carry",
    "dividend_yield_used",
    "dividend_yield_source",
    "risk_free_rate_used",
    "india_vix",
    "lot_size",
    "underlying_day_open",
    "underlying_day_high",
    "underlying_day_low",
    "underlying_prev_close",
    "price_source_for_iv",
    "data_quality_flag",
    # Signed put-call parity deviation in index points (positive == calls
    # rich), on both legs of a checked pair. NULL when the pair could not be
    # checked. See run_fetch.flag_parity_violations().
    "parity_deviation",
    # poller_id: which poller produced this row (Phase 1). Static
    # "gh-actions" for now, env-overridable via POLLER_ID -- lets multiple
    # pollers write into the same schema without colliding on identity.
    "poller_id",
]

# Kept as its own literal list rather than being defined as
# schema.column_names(): two independent lists plus schema's
# schema_matches_main_columns() guard catches drift, whereas deriving one from
# the other would make that guard vacuously true. tests/test_schema.py asserts
# they agree.


# ---------------------------------------------------------------------------
# Filenames and layout
# ---------------------------------------------------------------------------

def date_stamp(d) -> str:
    """YYYY-MM-DD, zero-padded, for filenames -- chosen so files sort
    correctly as plain text (lexicographic order == chronological order)."""
    return d.strftime("%Y-%m-%d")


def _day_str(day) -> str:
    """Accepts a date/datetime OR an already-formatted YYYY-MM-DD string, so
    callers that got their day from argparse (compact.py) and callers that got
    it from a clock (run_fetch.py) can both use the layout helpers."""
    if isinstance(day, str):
        return day
    return date_stamp(day)


# Everything outside this class becomes a single "-". Note that "." is
# deliberately NOT allowed: that makes ".." unrepresentable, so an
# env-supplied POLLER_ID can never escape the day directory, and it keeps the
# ".parquet" extension unambiguous.
_POLLER_SLUG_DISALLOWED = re.compile(r"[^A-Za-z0-9_-]+")
POLLER_SLUG_MAX_LEN = 32


def _slug_poller_id(poller_id) -> str:
    """
    Filename-safe form of a poller id. POLLER_ID is env-overridable, so an
    unsanitized value would be both a path-traversal vector
    (POLLER_ID=../../etc/foo) and a cross-platform filename hazard. Falls back
    to schema.DEFAULT_POLLER_ID if nothing printable survives.

    Only the FILENAME carries the slug; the poller_id *column* keeps whatever
    was configured, verbatim.
    """
    slug = _POLLER_SLUG_DISALLOWED.sub("-", str(poller_id or "")).strip("-")
    slug = slug[:POLLER_SLUG_MAX_LEN].strip("-")  # re-strip: the cut may end on "-"
    return slug or schema.DEFAULT_POLLER_ID


def shard_stamp(dt) -> str:
    """'2026-09-12T093015' -- the fixed-width (17 char) timestamp component of
    a shard filename. Kept poller-free on purpose: shard_path() appends the
    poller, so the fixed-width-prefix property lives in exactly one place.
    'T' rather than ':' keeps the name legal on a Windows checkout."""
    return f"{date_stamp(dt)}T{dt.strftime('%H%M%S')}"


def shard_dir(symbol: str, day) -> str:
    """vault/tables/<SYMBOL>/<YYYY-MM-DD> -- the directory holding one day's
    shards. Its absence/presence is the compaction signal."""
    return os.path.join(TABLES_DIR, symbol, _day_str(day))


def shard_path(symbol: str, fetch_ts_ist: datetime, poller_id=None) -> str:
    """
    The deterministic shard path for one fetch cycle. Requires a full
    datetime (not a date): the day directory and the time component both come
    from the same clock, so the name is internally consistent.

    poller_id=None means "use the module-level POLLER_ID".
    """
    slug = _slug_poller_id(POLLER_ID if poller_id is None else poller_id)
    return os.path.join(
        shard_dir(symbol, fetch_ts_ist.date()),
        f"MAIN-{shard_stamp(fetch_ts_ist)}-{slug}.parquet",
    )


def compacted_path(symbol: str, day) -> str:
    """vault/tables/<SYMBOL>/MAIN-<YYYY-MM-DD>.parquet -- one level up from the
    shards, named exactly like the pre-Phase-3 daily file."""
    return os.path.join(TABLES_DIR, symbol, f"MAIN-{_day_str(day)}.parquet")


def list_shards(symbol: str, day) -> list:
    """
    Sorted (== chronological, see module docstring) shard paths for one
    symbol-day; [] if the day directory doesn't exist.

    The glob is scoped to the day directory and anchored on ".parquet", which
    means it cannot reach the compacted file or the legacy MAIN-*.csv files at
    the symbol level, and it cannot see a writer's in-flight "*.parquet.tmp".
    """
    return sorted(glob.glob(os.path.join(shard_dir(symbol, day), "MAIN-*.parquet")))


def list_shard_days(symbol: str) -> list:
    """Sorted YYYY-MM-DD strings that currently have a shard directory, i.e.
    the days that still need compacting. Lets compact.py default to 'every
    uncompacted day' without being told."""
    symbol_dir = os.path.join(TABLES_DIR, symbol)
    if not os.path.isdir(symbol_dir):
        return []
    days = []
    for name in sorted(os.listdir(symbol_dir)):
        if not os.path.isdir(os.path.join(symbol_dir, name)):
            continue
        try:
            datetime.strptime(name, "%Y-%m-%d")
        except ValueError:
            continue  # not a day directory; ignore quietly
        days.append(name)
    return days


# ---------------------------------------------------------------------------
# Freshness check -- runs BEFORE any write
# ---------------------------------------------------------------------------

def freshness_ok(symbol: str, underlying_value, rows: list) -> bool:
    if not rows:
        print(f"  [{symbol}] freshness check FAILED: no rows produced.")
        return False
    if underlying_value in (None, 0):
        print(f"  [{symbol}] freshness check FAILED: missing/zero underlying value.")
        return False
    populated_price_rows = sum(
        1 for r in rows if r.get("ltp") or r.get("bid_price") or r.get("ask_price")
    )
    if populated_price_rows < max(1, len(rows) // 10):
        print(f"  [{symbol}] freshness check FAILED: "
              f"only {populated_price_rows}/{len(rows)} rows have any price data.")
        return False
    return True


# ---------------------------------------------------------------------------
# Raw JSON archive (per symbol, per day, gzip, day-appendable)
# ---------------------------------------------------------------------------

def append_raw_snapshot(symbol: str, day, snapshot: dict):
    """
    Appends one fetch-cycle's raw NSE responses to today's gzip archive for
    this symbol. Reads-modify-writes the whole file, which is fine at this
    data volume (a day's worth of 5-minute snapshots is small once gzipped).
    """
    out_dir = os.path.join(RAW_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"JSON-{date_stamp(day)}.json.gz")

    existing = []
    if os.path.isfile(out_path):
        try:
            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception as exc:  # noqa: BLE001
            print(f"  [{symbol}] WARNING: could not read existing raw archive "
                  f"({exc}); starting a fresh list for today (old file backed up).")
            backup_path = out_path + ".corrupt"
            try:
                os.replace(out_path, backup_path)
            except OSError:
                pass
            existing = []

    existing.append(snapshot)

    tmp_path = out_path + ".tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(existing, f)
    os.replace(tmp_path, out_path)  # atomic-ish swap, avoids half-written files

    print(f"  [{symbol}] raw archive updated -> {out_path} "
          f"({len(existing)} fetch cycle(s) today)")


# ---------------------------------------------------------------------------
# MAIN table -- typed parquet, one immutable shard per fetch cycle.
#
# The rows handed over by run_fetch.process_symbol() are a mix of ISO strings,
# NSE-JSON numbers-as-strings, floats and None, so type INFERENCE is unsafe:
# a column that happens to be all-null this cycle would land as null-typed, a
# quantity that arrives as "1234" would land as a string, and expiry_date would
# stay a DD-Mon-YYYY string forever. Every column is therefore converted
# explicitly against schema.VAULT_SCHEMA before the table is built.
# ---------------------------------------------------------------------------

def _to_datetime(v):
    if isinstance(v, datetime):
        return v
    # run_fetch stamps these with .isoformat(), i.e. an explicit UTC/IST
    # offset; pyarrow then converts the instant into the field's declared zone.
    return datetime.fromisoformat(str(v))


def _to_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    try:
        return datetime.strptime(s, "%d-%b-%Y").date()  # NSE's expiryDates format
    except ValueError:
        return date.fromisoformat(s)


def _to_int(v):
    if isinstance(v, bool):
        raise TypeError("bool is not an int64 quantity")
    return int(float(v))  # via float so "1234" and 1234.0 both land correctly


def _to_float(v):
    f = float(v)
    if not math.isfinite(f):
        # NaN/inf are stored as NULL rather than propagated: a single
        # pathological greek would otherwise poison any downstream mean().
        return None
    return f


def _to_string(v):
    return str(v)


def _converter_for(field):
    t = field.type
    if pa.types.is_timestamp(t):
        return _to_datetime
    if pa.types.is_date(t):
        return _to_date
    if pa.types.is_integer(t):
        return _to_int
    if pa.types.is_floating(t):
        return _to_float
    if pa.types.is_string(t):
        return _to_string
    raise TypeError(f"no converter for {field.name}: {t}")


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _build_table(rows: list):
    """
    Converts rows -> a pa.Table typed exactly as VAULT_SCHEMA.

    Returns (table, dropped_row_count, {column: conversion_failure_count}).

    Two kinds of problem are handled, both by counting rather than raising --
    one bad leg in an NSE response must not cost the symbol its whole cycle:

    - A value that won't convert becomes NULL and is counted.
    - A row that ends up NULL in any NON-NULLABLE column is DROPPED, because
      pyarrow treats nullability as metadata and will write the null happily.
      Realistically this only fires on expiry_date (run_fetch already skips
      unparseable strikes and always sets the other five identity columns),
      i.e. an NSE leg with no expiry -- which carries no usable information
      anyway (no T => no greeks) and is still preserved verbatim in the raw
      gzip archive. NOTE: the old CSV writer wrote such a row with an empty
      field instead of dropping it.
    """
    cols = {}
    failures = {}
    for field in schema.VAULT_SCHEMA:
        convert = _converter_for(field)
        out = []
        for row in rows:
            v = row.get(field.name)
            if _is_blank(v):
                out.append(None)
                continue
            try:
                out.append(convert(v))
            except (TypeError, ValueError, OverflowError):
                out.append(None)
                failures[field.name] = failures.get(field.name, 0) + 1
        cols[field.name] = out

    bad_indices = set()
    for name in schema.non_nullable_column_names():
        for i, v in enumerate(cols[name]):
            if v is None:
                bad_indices.add(i)

    if bad_indices:
        keep = [i for i in range(len(rows)) if i not in bad_indices]
        cols = {name: [values[i] for i in keep] for name, values in cols.items()}

    table = pa.Table.from_pydict(cols, schema=schema.VAULT_SCHEMA)
    return table, len(bad_indices), failures


def write_main_shard(symbol: str, fetch_ts_ist: datetime, rows: list,
                     poller_id=None) -> str:
    """
    Writes ONE fetch cycle's rows as a single typed parquet shard and returns
    its path (None if there was nothing to write).

    Never opens, reads, or rewrites any other shard -- including this symbol's
    earlier shards for the same day. The only file this function can modify is
    the one shard its own (symbol, fetch_ts, poller_id) names, so concurrent or
    retried cycles cannot corrupt each other's output.

    The write goes to "<path>.tmp" in the same directory and is then
    os.replace()d into place -- the same atomic-swap idiom append_raw_snapshot()
    uses above. A reader therefore sees either no shard or a complete one, and
    the ".tmp" name is outside list_shards()' glob for the same reason.
    """
    if not rows:
        return None

    resolved_poller = POLLER_ID if poller_id is None else poller_id
    out_path = shard_path(symbol, fetch_ts_ist, resolved_poller)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    table, dropped, failures = _build_table(rows)

    if dropped:
        print(f"::warning::[{symbol}] dropped {dropped} of {len(rows)} row(s) missing a "
              f"required identity column ({', '.join(schema.non_nullable_column_names())}); "
              f"the raw archive still has them verbatim.")
    if failures:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(failures.items()))
        print(f"::warning::[{symbol}] stored NULL for value(s) that would not convert "
              f"to their declared type: {detail}")

    # The filename's poller and the rows' poller_id column must describe the
    # same poller, or the archive's provenance is a lie. Warn rather than
    # raise: run_fetch's call site is not wrapped in a try/except, so raising
    # here would cost the remaining symbols their writes for a metadata
    # mismatch that doesn't corrupt anything.
    row_pollers = {r.get("poller_id") for r in rows}
    if row_pollers - {resolved_poller}:
        print(f"::warning::[{symbol}] shard filename says poller "
              f"'{_slug_poller_id(resolved_poller)}' but rows carry {sorted(row_pollers)}; "
              f"the poller_id column is authoritative for provenance.")

    if table.num_rows == 0:
        print(f"::warning::[{symbol}] no writable rows survived type coercion; "
              f"no shard written for {shard_stamp(fetch_ts_ist)}.")
        return None

    tmp_path = out_path + ".tmp"
    try:
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, out_path)  # atomic swap; overwrites its own shard on a re-run
    except Exception:
        # Don't leave a partial ".tmp" for the workflow's `git add` to commit.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(f"  [{symbol}] MAIN shard written -> {out_path} ({table.num_rows} rows)")
    return out_path
