"""
schema.py

The single, explicit Arrow schema for a MAIN table row. This is the contract
that the storage layer builds on -- the Phase 3 parquet writer, the Phase 3
compactor, and the Phase 6 rebuild-from-raw script all import VAULT_SCHEMA so
that types stay consistent across shards, compacted files, and any rebuilt
tables. Defining it correctly in Phase 1, before any parquet existed, was
deliberate: retrofitting column types after parquet files exist is expensive.

Relationship to vault_io.MAIN_COLUMNS:
- The field order here matches MAIN_COLUMNS, followed by poller_id (also
  appended to MAIN_COLUMNS in Phase 1). schema_matches_main_columns() asserts
  they stay in sync -- tests/test_schema.py calls it.
- MAIN_COLUMNS is deliberately kept as its own literal list rather than being
  defined as column_names(): two independent lists plus a guard catches drift,
  one list derived from the other makes the guard vacuously true.
- Phase 3's writer applies these types (ISO strings -> timestamp, DD-Mon-YYYY
  -> date32) in vault_io._build_table(); this module only declares them.

Type/nullability choices:
- Identity columns (timestamps, symbol, expiry, strike, option_type, poller_id)
  are non-nullable: a row without them is meaningless.
- Everything derived or quote-sourced is nullable, because the guardrails in
  run_fetch.py legitimately leave Greeks/prices/IV null on bad-quote or
  no-IV rows.
- Timestamps carry timezone: fetch_ts_utc in UTC, fetch_ts_ist in Asia/Kolkata,
  matching how run_fetch.py produces them.
- expiry_date is date32 (no intraday component); time_to_expiry_years carries
  the fractional-year precision instead.
"""

import pyarrow as pa

POLLER_ID_COLUMN = "poller_id"
DEFAULT_POLLER_ID = "gh-actions"


def _field(name, type_, nullable=True):
    return pa.field(name, type_, nullable=nullable)


VAULT_SCHEMA = pa.schema([
    # --- identity / timestamps (non-null) ---
    _field("fetch_ts_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    _field("fetch_ts_ist", pa.timestamp("us", tz="Asia/Kolkata"), nullable=False),
    _field("symbol", pa.string(), nullable=False),
    _field("expiry_date", pa.date32(), nullable=False),
    _field("strike", pa.float64(), nullable=False),
    _field("option_type", pa.string(), nullable=False),

    # --- quote data (nullable) ---
    _field("underlying_value", pa.float64()),
    _field("bid_price", pa.float64()),
    _field("bid_qty", pa.int64()),
    _field("ask_price", pa.float64()),
    _field("ask_qty", pa.int64()),
    # Whole-book aggregate depth, as opposed to the level-1 bid_qty/ask_qty
    # above. Nullable: NSE omits them on some legs, and a missing field must
    # stay NULL rather than become a fabricated 0.
    _field("total_buy_quantity", pa.int64()),
    _field("total_sell_quantity", pa.int64()),
    _field("ltp", pa.float64()),
    _field("mid_price", pa.float64()),
    _field("open_interest", pa.int64()),
    _field("change_in_oi", pa.int64()),
    _field("total_traded_volume", pa.int64()),
    _field("pchange_vs_prev_close", pa.float64()),
    _field("nse_iv", pa.float64()),

    # --- greeks (nullable; left null on no-IV / bad-quote rows) ---
    _field("delta", pa.float64()),
    _field("gamma", pa.float64()),
    _field("theta", pa.float64()),
    _field("vega", pa.float64()),
    _field("vanna", pa.float64()),
    _field("charm", pa.float64()),
    _field("vomma", pa.float64()),
    _field("speed", pa.float64()),
    _field("zomma", pa.float64()),
    _field("color", pa.float64()),
    _field("veta", pa.float64()),
    _field("omega", pa.float64()),
    _field("dual_delta", pa.float64()),
    _field("dual_gamma", pa.float64()),

    # --- pricing inputs / provenance (nullable) ---
    _field("time_to_expiry_years", pa.float64()),
    _field("futures_price", pa.float64()),
    _field("implied_cost_of_carry", pa.float64()),
    _field("dividend_yield_used", pa.float64()),
    _field("dividend_yield_source", pa.string()),
    _field("risk_free_rate_used", pa.float64()),
    _field("india_vix", pa.float64()),
    _field("lot_size", pa.int64()),
    _field("underlying_day_open", pa.float64()),
    _field("underlying_day_high", pa.float64()),
    _field("underlying_day_low", pa.float64()),
    _field("underlying_prev_close", pa.float64()),
    _field("price_source_for_iv", pa.string()),
    _field("data_quality_flag", pa.string()),
    # Signed put-call parity deviation in index points, (C-P) - (S*e^-qT -
    # K*e^-rT); positive means calls rich. Written to BOTH legs of a checked
    # pair, un-negated. NULL when the pair was not checkable.
    _field("parity_deviation", pa.float64()),

    # --- poller identity (non-null, new in Phase 1) ---
    _field(POLLER_ID_COLUMN, pa.string(), nullable=False),
])


def column_names() -> list:
    return [f.name for f in VAULT_SCHEMA]


def non_nullable_column_names() -> list:
    """
    The identity columns a row is meaningless without. Driven off the schema
    itself so the writer's guard can never drift from the declaration.

    This exists because pyarrow treats nullability as *metadata*: it will
    happily write a null into a nullable=False field without complaining, so
    the constraint has to be enforced in Python before the table is built.
    See vault_io._build_table().
    """
    return [f.name for f in VAULT_SCHEMA if not f.nullable]


def schema_matches_main_columns(main_columns: list) -> bool:
    """
    True iff VAULT_SCHEMA's field order equals main_columns exactly. Guards
    against the schema and vault_io.MAIN_COLUMNS drifting apart. Pass
    vault_io.MAIN_COLUMNS in; tests/test_schema.py does exactly that.
    """
    return column_names() == list(main_columns)
