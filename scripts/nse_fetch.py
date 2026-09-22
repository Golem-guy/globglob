"""
nse_fetch.py

Plain `requests`-based client for NSE's public JSON endpoints -- no
nsepython/nsepythonserver dependency. This was a deliberate rewrite after
live debugging showed:

1. The classic option-chain endpoint (option-chain-indices) returns an
   empty {} even when hit directly, from BOTH a GitHub Actions cloud
   runner AND a residential IP -- it's very likely simply retired on
   NSE's side, not "blocked."
2. A newer endpoint (NextApi/apiClient/GetQuoteApi, functionName=
   getSymbolDerivativesData) DOES return real, live, full option-chain
   data -- confirmed with live output during debugging. nse_optionchain_
   scrapper() (from nsepython/nsepythonserver) apparently wraps this
   endpoint internally but its own reshaping logic was returning {}
   regardless -- a library bug, not an NSE/IP problem.
3. This project's own bhavcopy downloader (scraper.py) has been reliably
   pulling NSE data from GitHub Actions the whole time, using nothing
   more exotic than: a requests.Session(), a real browser User-Agent, and
   visiting nseindia.com first to collect cookies before hitting the
   actual target URL. This module ports that exact proven pattern to the
   live API endpoints instead of assuming a special cloud-workaround
   library was ever necessary.

IMPORTANT DATA LIMITATION: the working option-chain endpoint above does
NOT include bid/ask (no live two-sided quote) or NSE's own published IV.
Both were available from the old (now-dead) endpoint. This means:
- IV/Greeks are solved from LTP only -- there's no mid-price to prefer
  over it anymore, since there's no bid/ask at all.
- The below-intrinsic-value guardrail in greeks.implied_vol() is now the
  ONLY protection against a bad price -- it still works fine (only needs
  LTP), it just can't distinguish "thin/wide quote" from "no quote"
  anymore, since there's no quote to inspect.
- nse_iv is always None going forward -- the cross-check column has
  nothing to cross-check against with this data source.
See run_fetch.py and the README data dictionary for how this is reflected
in the output columns.
"""

import threading
import time
from datetime import datetime

import requests

import config_loader

MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 3  # 3s, 6s, 12s...

# Symbols are fetched CONCURRENTLY (see run_fetch.py's ThreadPoolExecutor), so
# multiple threads reach print() at once. Route every log line through this
# lock so two threads' output can't interleave into one garbled line.
# run_fetch's gha_warning/gha_error also go through log() -> the same lock.
_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(msg)


BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

WARMUP_URL = "https://www.nseindia.com/option-chain"

OPTION_CHAIN_URL = (
    "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
    "?functionName=getSymbolDerivativesData&symbol={symbol}"
)
OPTION_CHAIN_V3_URL = (
    "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={expiry}"
)
ALL_INDICES_URL = "https://www.nseindia.com/api/allIndices"

# quote-derivative is CONFIRMED DEAD -- every URL shape tried against it
# (bare symbol, symbol+identifier, identifier only) returned a genuine
# HTTP 404 "Resource not found" during live debugging, not an empty
# response or a block. It used to be this pipeline's source for both
# futures prices AND lot size. Futures now come from
# parse_futures_from_entries() (see above, uses the option-chain response
# itself). Lot size has NO live source at all anymore -- the
# lot_size_fallback in config/symbols.yaml is the only source, full stop,
# until/unless a replacement endpoint is found.

# Lot sizes, index display names, and the India VIX display name all now
# live in config/symbols.yaml, read via config_loader -- they used to be
# hardcoded here. Lot size still has NO live source (the quote-derivative
# endpoint is dead), so the config's lot_size_fallback is the only source;
# re-check https://www.nseindia.com/all-reports-derivatives periodically and
# update symbols.yaml when NSE next revises lot sizes, since there's no
# live endpoint to catch a revision automatically.

_session = None  # module-level, lazily created, reused across all calls in a run
_session_lock = threading.Lock()  # guards the one-time session creation below


def _get_session() -> requests.Session:
    """
    Creates (once per script run) or returns the shared session. Warms it
    up by visiting the option-chain page first, exactly like the bhavcopy
    downloader visits the homepage first -- this is what actually
    populates the cookies NSE's API checks for.

    Thread-safe: symbols are fetched concurrently (see run_fetch.py), so the
    creation is guarded by _session_lock with a double-checked `is not None`
    -- at most ONE thread ever builds+warms the session, the rest reuse it.
    In practice run_fetch calls warmup_session() once BEFORE spawning any
    threads, so the lock is uncontended; the locking is belt-and-suspenders
    in case _get_session() is ever first reached from inside a thread.

    IMPORTANT: the warm-up is wrapped in its own try/except and always
    results in a cached session, even on failure. Without this, a single
    slow/failed warm-up would silently retry on EVERY subsequent call in
    the same run (since _session would stay None), with no logging and
    no retry cap -- eating potentially minutes of runtime with zero
    visibility. Better to log it once, proceed without fresh cookies, and
    let the actual API calls' own retry logic handle any consequences.
    """
    global _session
    if _session is not None:
        return _session

    with _session_lock:
        if _session is not None:  # another thread built it while we waited
            return _session
        session = requests.Session()
        session.headers.update(BASE_HEADERS)
        try:
            session.get(WARMUP_URL, timeout=15)
        except Exception as exc:  # noqa: BLE001
            log(f"  [warmup] WARNING: session warm-up request failed ({exc}); "
                f"proceeding without fresh cookies for this run.")
        _session = session  # cache regardless, so this can't retry unboundedly
        return _session


def warmup_session() -> requests.Session:
    """Create + warm the shared session ONCE, explicitly, before any worker
    threads start. The lazy path in _get_session() is thread-safe too (see
    there), but doing it up front makes the race impossible and keeps the
    warm-up log line from interleaving with per-symbol fetch logs."""
    return _get_session()


def _retry(fn, *args, what: str, **kwargs):
    """Generic retry wrapper with exponential backoff. Logs how long the call
    actually took -- makes a slow-but-not-failing NSE response visible in the
    logs instead of just showing up as unexplained total run duration.

    There is deliberately NO fixed pause after a successful call anymore: the
    old flat SLEEP_BETWEEN_CALLS_SECONDS sat after every success (the last call
    of the run included) and, now that symbols run concurrently, added latency
    for no benefit. The exponential backoff on FAILURE below is preserved --
    that's the part that actually protects against hammering a struggling NSE.
    Logs via log() (lock-guarded) since symbols are fetched concurrently."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.monotonic() - start
            log(f"  [{what}] ok in {elapsed:.1f}s")
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                log(f"  [{what}] attempt {attempt}/{MAX_RETRIES} failed after "
                    f"{elapsed:.1f}s ({exc}); retrying in {wait}s...")
                time.sleep(wait)
    raise last_exc


def _get_json(url: str, what: str, referer: str = None) -> dict:
    session = _get_session()
    headers = {"Referer": referer} if referer else {}

    def _do_request():
        resp = session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    return _retry(_do_request, what=what)


# ---------------------------------------------------------------------------
# Option chain (flat per-contract entries -- see module docstring)
# ---------------------------------------------------------------------------

def fetch_option_chain(symbol: str) -> dict:
    """
    Returns the raw response: {"data": [...flat entries...], "timestamp": ...}.
    Each entry in "data" is ONE contract leg. Most are options (CE/PE), but
    a handful are the underlying's own FUTURES contracts, distinguished by
    instrumentType starting with "FUT" (e.g. FUTIDX for index futures,
    presumably FUTSTK for stock futures) rather than "OPTIDX"/"OPTSTK".
    This means a single call covers both options AND futures -- see
    parse_futures_from_entries() below, which replaces what used to be a
    separate call to the (now confirmed dead, 404s on every URL shape
    tried) quote-derivative endpoint.
    """
    url = OPTION_CHAIN_URL.format(symbol=symbol)
    return _get_json(url, what=f"option_chain:{symbol}", referer=WARMUP_URL)


def flatten_v3_entries(v3_raw: dict) -> list:
    """
    Converts option-chain-v3's nested per-strike {CE: {...}, PE: {...}}
    structure into a flat list of one dict per option leg, tagged with
    option_type and a consistent-format expiry (the per-strike-entry
    "expiryDates" field, DD-Mon-YYYY -- NOT the inner leg's own
    "expiryDate" field, which uses a different DD-MM-YYYY format on this
    endpoint; using the outer one keeps expiry formatting consistent with
    every other date field in this pipeline).
    """
    records = (v3_raw or {}).get("records", {})
    entries = records.get("data", [])
    flat = []
    for entry in entries:
        strike = entry.get("strikePrice")
        expiry_outer = entry.get("expiryDates")
        for opt_type in ("CE", "PE"):
            leg = entry.get(opt_type)
            if not leg:
                continue
            leg = dict(leg)
            leg["option_type"] = opt_type
            leg["strikePrice"] = strike
            leg["expiryDates"] = expiry_outer
            flat.append(leg)
    return flat


def parse_futures_from_entries(entries: list) -> dict:
    """
    Returns {expiry_date_str: futures_price}, built from the SAME entries
    list returned by fetch_option_chain() -- filtering for futures
    contracts rather than a separate API call. Confirmed live: NIFTY's
    response includes ~3 futures entries (near/next/far month) alongside
    the option entries, with strikePrice "0.00" and optionType "XX" as
    placeholders (they're not real options, just tagged the same way).
    """
    futures = {}
    for entry in entries:
        instrument = entry.get("instrumentType", "")
        if instrument.startswith("FUT"):
            expiry = entry.get("expiryDate")
            ltp = entry.get("lastPrice")
            if expiry and ltp:
                futures[expiry] = ltp
    return futures


# ---------------------------------------------------------------------------
# option-chain-v3: the endpoint with real bid/ask and NSE's own IV.
#
# CONFIRMED LIVE: requires an explicit &expiry= param -- omitting it
# returns HTTP 200 but completely empty records (no data, no metadata at
# all). When a VALID expiry is supplied, the response also includes the
# FULL list of every expiry NSE has for that symbol under
# records.expiryDates -- so one call with any known-good expiry doubles
# as a way to discover every other valid expiry.
#
# Two-step fetch, per symbol, per cycle:
#   1. Bootstrap: use a guaranteed-valid expiry string from
#      fetch_option_chain() (the flat endpoint, already proven reliable)
#      to make the first v3 call. This also hands back the full expiry list.
#   2. Targets: from that list, pick BOTH the nearest upcoming expiry
#      (pick_nearest_weekly_expiry) and the nearest monthly
#      (pick_nearest_monthly_expiry), de-duplicate them, and fetch only the
#      ones the bootstrap call has not already returned. In expiry week the
#      two picks are the same date, and the bootstrap expiry is often one of
#      them, so the extra calls range from 0 to 2 -- see the cache in
#      run_fetch.process_symbol() for exactly how that is decided.
# ---------------------------------------------------------------------------

def fetch_option_chain_v3(symbol: str, expiry: str) -> dict:
    url = OPTION_CHAIN_V3_URL.format(symbol=symbol, expiry=expiry)
    return _get_json(url, what=f"option_chain_v3:{symbol}:{expiry}", referer=WARMUP_URL)


def _parse_expiry_dates(expiry_dates: list) -> list:
    """
    [(date, original_string), ...] for every entry that parses as NSE's
    DD-Mon-YYYY, sorted ascending; unparseable/None entries are dropped.

    Shared by both pickers below so they can never disagree about what counts
    as parseable or about ordering. They are asked the same question about the
    same list on every cycle, and a divergence between them would surface as a
    silently wrong expiry being archived rather than as an error.
    """
    parsed = []
    for d in expiry_dates or []:
        try:
            dt = datetime.strptime(d, "%d-%b-%Y").date()
        except (ValueError, TypeError):
            continue
        parsed.append((dt, d))
    parsed.sort()
    return parsed


def pick_nearest_weekly_expiry(expiry_dates: list, as_of_date) -> str:
    """
    The soonest expiry that has not passed as_of_date -- "weekly" in the loose
    sense of "the front contract", NOT in the sense of "a contract NSE labels
    weekly". NSE tags neither kind, and the only distinction is the convention
    pick_nearest_monthly_expiry() encodes: whichever date falls last in its
    month is that month's monthly, and the earlier ones are weeklies.

    In expiry week the front contract IS that month's monthly, so this
    returning the same string as pick_nearest_monthly_expiry() is CORRECT, not
    a bug. De-duplicating the two picks (and therefore not fetching the same
    chain twice) is the caller's job -- see run_fetch.process_symbol().

    as_of_date itself still counts as upcoming, matching the monthly picker's
    >= comparison: contracts trade until 15:30 on their own expiry day.

    Returns None if expiry_dates is empty or nothing parses.
    """
    parsed = _parse_expiry_dates(expiry_dates)
    if not parsed:
        return None

    for dt, original in parsed:
        if dt >= as_of_date:
            return original
    return parsed[-1][1]  # everything's in the past (shouldn't happen); best effort


def pick_nearest_monthly_expiry(expiry_dates: list, as_of_date) -> str:
    """
    NSE lists both weekly and monthly expiries together with no explicit
    tag distinguishing them. The monthly contract for a given calendar
    month is, by convention, the LAST (latest) expiry date that falls in
    that month -- weeklies fill in the earlier Tuesdays. This groups the
    given expiry_dates by (year, month), takes the latest date in each
    group as that month's "monthly," and returns the nearest one that
    hasn't already passed as_of_date.

    Returns None if expiry_dates is empty or nothing parses.
    """
    parsed = _parse_expiry_dates(expiry_dates)
    if not parsed:
        return None

    monthly_by_year_month = {}
    for dt, original in parsed:
        key = (dt.year, dt.month)
        if key not in monthly_by_year_month or dt > monthly_by_year_month[key][0]:
            monthly_by_year_month[key] = (dt, original)

    monthlies = sorted(monthly_by_year_month.values())
    for dt, original in monthlies:
        if dt >= as_of_date:
            return original
    return monthlies[-1][1]  # everything's in the past (shouldn't happen); best effort


# ---------------------------------------------------------------------------
# Index snapshot (day OHLC for all indices + India VIX) -- confirmed
# reliable from both cloud and residential IPs throughout debugging.
# ---------------------------------------------------------------------------

def fetch_index_snapshot_raw() -> dict:
    return _get_json(ALL_INDICES_URL, what="all_indices")


def parse_index_snapshot(raw: dict, symbol: str) -> dict:
    target_name = config_loader.index_display_name(symbol)
    return _extract_index_row(raw, target_name)


def parse_india_vix(raw: dict) -> dict:
    return _extract_index_row(raw, config_loader.india_vix_display_name())


def _extract_index_row(raw: dict, target_name: str) -> dict:
    empty = {"open": None, "high": None, "low": None, "prev_close": None,
             "last": None, "dy": None}
    if not raw or not target_name:
        return empty
    for row in raw.get("data", []):
        name = (row.get("index") or row.get("indexSymbol") or "").strip().upper()
        if name == target_name.upper():
            return {
                "open": row.get("open"),
                "high": row.get("high") or row.get("dayHigh"),
                "low": row.get("low") or row.get("dayLow"),
                "prev_close": row.get("previousClose"),
                "last": row.get("last") or row.get("lastPrice"),
                # NSE's own published dividend yield for this index, as a
                # percentage string (e.g. "1.18" means 1.18%). This is the
                # PRIMARY source for the dividend yield fed into the
                # Greeks -- see run_fetch.py's get_dividend_yield_and_carry().
                # It's a slow-moving, NSE-published number, unlike a
                # per-cycle futures-basis calculation, which can blow up
                # to nonsensical values for near-expiry contracts (see
                # README "Data source history" for the concrete example
                # that motivated this switch).
                "dy": row.get("dy"),
            }
    return empty
