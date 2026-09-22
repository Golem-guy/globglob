"""
compact.py

Merges one symbol-day's per-cycle parquet shards into a single compacted file:

    vault/tables/<SYMBOL>/<YYYY-MM-DD>/MAIN-*.parquet   (many small shards)
        -> vault/tables/<SYMBOL>/MAIN-<YYYY-MM-DD>.parquet   (one file)

Run it manually, e.g. at the end of a trading day:

    python scripts/compact.py --symbol NIFTY --date 2026-09-12
    python scripts/compact.py --date 2026-09-12          # all configured symbols
    python scripts/compact.py --symbol NIFTY             # every uncompacted day
    python scripts/compact.py --dry-run ...              # show the plan, touch nothing

Deliberately NOT wired into .github/workflows/fetch-options.yml. A fetch cycle
must stay a pure append of one new shard; folding a destructive merge into the
same run would put deletion of good data behind the same 5-minute cron that is
already racing other runners to push.

WHY THE SAFETY DANCE BELOW EXISTS
---------------------------------
Compaction is the only operation in this repo that DELETES archived data, so
every failure mode has to leave the shards untouched. The order is fixed:

    merge -> write to a temp file -> verify the temp file -> atomically
    rename it into place -> and only then delete the shards.

Nothing is deleted until a verified, complete replacement is already on disk
under its final name. Every failure before the rename leaves the shards
byte-identical and the target absent; the one window after it (rename done,
shards not yet deleted) is detected and resumed on the next run -- see
_resume_or_refuse().

WHY THE ARROW HOP AND THE EXPLICIT CAST
---------------------------------------
The obvious implementation is DuckDB's `COPY (...) TO 'out.parquet'`. It is
wrong here, subtly: DuckDB's TIMESTAMPTZ is a bare instant with no per-value
zone, so fetch_ts_ist round-trips through DuckDB as timestamp[us, tz=UTC].
The instant is preserved, but the stored zone METADATA silently drifts away
from VAULT_SCHEMA, and compacted files stop being schema-identical to shards.
So the merge goes DuckDB -> Arrow -> cast(VAULT_SCHEMA) -> pyarrow write. The
cast is metadata-only for the timestamps (same instant, declared zone
restored) and is load-bearing, not cosmetic. A symbol-day is ~10k rows, so
holding it in memory costs nothing.
"""

import argparse
import os
import sys

import duckdb
import pyarrow.parquet as pq

import config_loader
import schema
import vault_io

# Deterministic row order inside a compacted file: chronological first, then a
# stable key within a cycle. Makes compacted files reproducible and gives
# readers a useful sort order for free.
MERGE_ORDER_BY = "fetch_ts_utc, expiry_date, strike, option_type"


class CompactionError(Exception):
    """A symbol-day could not be compacted. Raised only from points where the
    original shards are still intact."""


def _sql_file_list(paths) -> str:
    """DuckDB list literal of file paths, single-quotes escaped. Built as a
    literal rather than a bound parameter so this works identically across
    DuckDB versions' handling of list parameters in read_parquet()."""
    return "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


def _connect():
    con = duckdb.connect()
    # Pin the session zone so TIMESTAMPTZ values come back identically on every
    # machine. The final cast to VAULT_SCHEMA restores Asia/Kolkata on
    # fetch_ts_ist regardless, but determinism here keeps comparisons honest.
    con.execute("SET TimeZone='UTC'")
    return con


def _shard_row_counts(shards: list) -> dict:
    """Rows per shard, read from each parquet FOOTER -- metadata only, no scan
    of the data pages."""
    return {p: pq.ParquetFile(p).metadata.num_rows for p in shards}


def _distinct_fetch_ts(con, paths: list) -> set:
    """
    The distinct fetch instants in `paths`, as integer microseconds since the
    epoch.

    epoch_us() rather than the bare TIMESTAMPTZ on purpose. This set is only
    ever used for an identity comparison (are the shards a subset of the
    target?), and handing DuckDB's TIMESTAMPTZ back to Python routes it
    through duckdb's timezone machinery, which requires pytz to be installed
    and raises InvalidInputException when it isn't. An integer instant needs
    no zone, compares exactly, and cannot drift.
    """
    rows = con.execute(
        f"SELECT DISTINCT epoch_us(fetch_ts_utc) FROM read_parquet({_sql_file_list(paths)})"
    ).fetchall()
    return {r[0] for r in rows}


def _delete_shards(symbol: str, day, shards: list):
    """Removes the shard files, then the day directory. Called ONLY after a
    verified compacted file is in place under its final name."""
    for path in shards:
        os.unlink(path)
    day_dir = vault_io.shard_dir(symbol, day)
    try:
        os.rmdir(day_dir)
    except OSError as exc:  # non-shard files present; leave it and say so
        print(f"::warning::[{symbol} {vault_io._day_str(day)}] shards deleted but the day "
              f"directory is not empty, leaving it in place ({exc}).")


def _merge_to_temp(con, sources: list, tmp_path: str, distinct: bool, expected: int):
    """
    Merges `sources` into tmp_path and VERIFIES the written file before
    returning. Raises (after unlinking the temp) if verification fails --
    which is why this is the last step before the rename: everything that can
    go wrong has gone wrong by the time we return.
    """
    select = "SELECT DISTINCT *" if distinct else "SELECT *"
    query = (f"{select} FROM read_parquet({_sql_file_list(sources)}, union_by_name=true) "
             f"ORDER BY {MERGE_ORDER_BY}")

    # to_arrow_table(), NOT .arrow(): as of duckdb 1.5 the latter hands back a
    # streaming pyarrow.RecordBatchReader, which has neither .select() nor
    # .cast(). (fetch_arrow_table() is the same thing under its deprecated
    # older name.) A symbol-day is ~10k rows, so materialising costs nothing.
    table = con.execute(query).to_arrow_table()
    # Select by name before casting: makes the cast independent of the column
    # ORDER DuckDB happens to produce, and fails loudly if a column is missing.
    table = table.select(schema.column_names()).cast(schema.VAULT_SCHEMA)

    pq.write_table(table, tmp_path, compression="zstd")

    # Verify off the FILE, not off the in-memory table -- the point is to
    # prove that what actually landed on disk is complete and correctly typed.
    written = pq.ParquetFile(tmp_path)
    if written.metadata.num_rows != expected:
        raise CompactionError(
            f"row count mismatch: merged file has {written.metadata.num_rows} rows, "
            f"expected {expected}. Shards left untouched."
        )
    if not written.schema_arrow.equals(schema.VAULT_SCHEMA, check_metadata=False):
        raise CompactionError(
            "merged file's schema does not match VAULT_SCHEMA. Shards left untouched."
        )


def _resume_or_refuse(con, symbol: str, day, shards: list, target: str, force: bool):
    """
    Called when the compacted target ALREADY exists. Decides between finishing
    an interrupted run and refusing.

    The hazard being defended against: a previous run renamed the merged file
    into place and then died before deleting the shards. Blindly merging
    target + shards on the next run would double-count every row. So compare
    identities instead of trusting file existence -- if every fetch_ts in the
    shards is already represented in the target, the merge did happen and only
    the cleanup is outstanding.

    Returns "resumed" if it completed that cleanup; raises otherwise.
    """
    label = f"{symbol} {vault_io._day_str(day)}"
    target_ts = _distinct_fetch_ts(con, [target])
    shard_ts = _distinct_fetch_ts(con, shards)

    if shard_ts <= target_ts:
        print(f"  [{label}] already compacted: every shard fetch_ts is present in "
              f"{target}. Finishing the interrupted cleanup.")
        _delete_shards(symbol, day, shards)
        return "resumed"

    missing = len(shard_ts - target_ts)
    if not force:
        raise CompactionError(
            f"{target} already exists but {missing} shard fetch_ts value(s) are NOT in "
            f"it, so this is not a resumable interrupted run. Refusing to merge, because "
            f"a blind union would double-count the overlapping rows. Inspect both, then "
            f"re-run with --force to rebuild the target from (target UNION shards) with "
            f"full-row de-duplication."
        )
    return "force"


def compact_symbol_day(symbol: str, day, force: bool = False,
                       dry_run: bool = False) -> str:
    """
    Compacts one symbol-day. Returns a short status string
    ("nothing" | "dry-run" | "resumed" | "compacted"); raises CompactionError
    (or an OSError from the filesystem) on failure, always from a point where
    the shards are still intact.
    """
    day_str = vault_io._day_str(day)
    label = f"{symbol} {day_str}"

    shards = vault_io.list_shards(symbol, day)
    if not shards:
        print(f"  [{label}] nothing to compact (no shards).")
        return "nothing"

    counts = _shard_row_counts(shards)
    expected = sum(counts.values())
    target = vault_io.compacted_path(symbol, day)

    print(f"  [{label}] {len(shards)} shard(s), {expected} rows -> {target}")
    for path in shards:
        print(f"      {os.path.basename(path)}  {counts[path]} rows")

    con = _connect()
    try:
        distinct = False
        if os.path.exists(target):
            outcome = _resume_or_refuse(con, symbol, day, shards, target, force)
            if outcome == "resumed":
                return "resumed"
            # --force: rebuild from target UNION shards, de-duplicated, and
            # verify against the DISTINCT count rather than the plain sum.
            sources = [target] + shards
            distinct = True
            expected = con.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM "
                f"read_parquet({_sql_file_list(sources)}, union_by_name=true))"
            ).fetchone()[0]
            print(f"  [{label}] --force: merging target + shards, "
                  f"{expected} distinct row(s) expected.")
        else:
            sources = shards

        if dry_run:
            print(f"  [{label}] --dry-run: nothing written, nothing deleted.")
            return "dry-run"

        os.makedirs(os.path.dirname(target), exist_ok=True)
        # Same directory as the target => same filesystem => os.replace() is a
        # genuine atomic rename, not a copy.
        tmp_path = target + ".tmp"
        try:
            _merge_to_temp(con, sources, tmp_path, distinct=distinct, expected=expected)
            os.replace(tmp_path, target)          # <- the point of no return
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        _delete_shards(symbol, day, shards)
        print(f"  [{label}] compacted {expected} rows -> {target}; "
              f"{len(shards)} shard(s) removed.")
        return "compacted"
    finally:
        con.close()


def _targets(symbols, date_str):
    """(symbol, day) pairs to process, from the CLI args and what's on disk."""
    for symbol in symbols:
        if date_str:
            yield symbol, date_str
            continue
        days = vault_io.list_shard_days(symbol)
        if not days:
            print(f"  [{symbol}] no uncompacted day directories found.")
        for day in days:
            yield symbol, day


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge a symbol-day's per-cycle MAIN parquet shards into one file.")
    parser.add_argument("--symbol", action="append", default=None,
                        help="Symbol to compact; repeatable. Default: every symbol in "
                             "config/symbols.yaml.")
    parser.add_argument("--date", default=None,
                        help="Day to compact, YYYY-MM-DD. Default: every day that still "
                             "has a shard directory.")
    parser.add_argument("--force", action="store_true",
                        help="When the compacted file already exists and the shards are "
                             "NOT a subset of it, rebuild it from (target UNION shards) "
                             "with full-row de-duplication.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan; write nothing, delete nothing.")
    args = parser.parse_args(argv)

    symbols = args.symbol or config_loader.all_symbol_names()
    failures = []

    for symbol, day in _targets(symbols, args.date):
        try:
            compact_symbol_day(symbol, day, force=args.force, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            # Keep going: one bad symbol-day must not stop the others, and by
            # construction its shards are still intact.
            failures.append((symbol, day, exc))
            print(f"::error::[{symbol} {day}] compaction failed: {exc}")

    if failures:
        print(f"::error::{len(failures)} symbol-day(s) failed to compact; "
              f"their shards are unchanged.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
