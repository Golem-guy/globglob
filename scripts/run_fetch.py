"""
run_fetch.py

Entry point. One run = one fetch cycle across all three symbols.

Design principles baked in here (per everything discussed building this):
- Skip entirely, cleanly, on weekends/holidays -- no files touched, exit 0.
- Every symbol is fetched and processed independently: one symbol's
  failure (NSE hiccup, parsing error, whatever) is logged and skipped,
  the other symbols still get processed and written.
- fetch_ts is the script's OWN clock, logged on every row -- never trust
  the nominal cron time, cron drift is expected and handled by this.
- IV/Greeks prefer mid_price (bid+ask)/2 over LTP; LTP can be a stale
  trade from hours ago on an illiquid strike, mid_price reflects live
  market-maker quotes even without a trade.
- Bad-quote guardrail: price below intrinsic value -> no IV/Greeks (not
  forced), flagged no_price_available or similar in data_quality_flag.
- Wide-quote flag: a usable price with a very wide bid-ask spread gets
  IV/Greeks computed, but is flagged wide_quote_low_liquidity so you can
  choose whether to trust it downstream.
- Freshness check runs BEFORE writing anything for a symbol.
- Cost-of-carry (and therefore dividend yield) is derived from the
  futures price for the same symbol/expiry, not assumed -- see
  get_cost_of_carry(). Falls back to a static assumption, flagged, when
  futures aren't available.
- Raw JSON (untouched NSE responses) is archived per symbol per day
  regardless of whether the MAIN row-building succeeds, since the raw
  archive is the future-proofing layer and shouldn't be gated on today's
  parsing logic being perfect.
"""

import concurrent.futures
import math
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import config_loader
import greeks
import holidays
import nse_fetch
import vault_io

IST = ZoneInfo("Asia/Kolkata")

# Which poller produced these rows. Static default for the GitHub Actions
# runner; env-overridable so a different poller (local, another CI) tags its
# rows distinctly without a code change.
#
# Re-exported from vault_io rather than read from the environment again here:
# since Phase 3 the shard FILENAME also carries the poller id, and the name
# must describe the same poller as the column. One read, one source.
POLLER_ID = vault_io.POLLER_ID

RISK_FREE_RATE = 0.065          # static assumption; only affects discounting
                                  # once cost-of-carry is futures-implied
FALLBACK_DIVIDEND_YIELD = 0.0    # used only when futures price unavailable
WIDE_QUOTE_SPREAD_PCT = 0.15     # bid-ask spread / mid_price threshold

# --- put-call parity check (Phase 5) ---
PARITY_FLAG = "parity_violation"
PARITY_RATE_DELTA = 0.005        # +/-50bp band around the row's own r
PARITY_TICK_FLOOR = 0.05         # one NSE index-option tick
PARITY_WARN_RATE = 0.10          # per-expiry violation rate earning a warning
PARITY_MIN_SAMPLE = 5            # ...but only once this many pairs were checked


def gha_warning(msg: str):
    # Routed through nse_fetch.log() so it shares the ONE print lock with the
    # concurrent per-symbol fetch logs -- process_symbol() runs inside worker
    # threads, so its warnings must not interleave with other threads' output.
    nse_fetch.log(f"::warning::{msg}")


def gha_error(msg: str):
    nse_fetch.log(f"::error::{msg}")


def years_to_expiry(expiry_str: str, as_of: datetime) -> float:
    """expiry_str like '25-Sep-2026'. Options stop trading at 15:30 IST."""
    expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y").replace(
        hour=15, minute=30, tzinfo=IST
    )
    delta = (expiry_dt - as_of).total_seconds()
    return max(delta, 0.0) / (365.0 * 24 * 3600)


def get_dividend_yield_and_carry(futures_by_expiry: dict, expiry_date: str,
                                  S: float, T: float, index_dy_pct):
    """
    Returns (q_used, dividend_yield_source, implied_cost_of_carry, futures_price).

    q_used is sourced from the underlying INDEX'S OWN published dividend
    yield (index_dy_pct, from allIndices' "dy" field) whenever available --
    a slow-moving, NSE-published number. This replaced an earlier design
    that derived q from the futures-basis formula (b = ln(F/S)/T): that
    approach blows up for near-expiry contracts, since dividing by a tiny
    T annualizes even normal basis noise into an extreme rate (confirmed
    live: a 4-day-to-expiry NIFTY strike showed an implied cost-of-carry
    of 20%+ and a dividend yield of -14%, which then distorted every
    Greek). The index's own published yield has no such T-dependency.

    futures_price / implied_cost_of_carry are still computed and returned
    when a futures price is available for this expiry -- purely as
    informational/diagnostic columns now, not used to derive q.
    """
    F = futures_by_expiry.get(expiry_date)
    b = None
    if F and S and T and T > 0:
        b = math.log(F / S) / T

    if index_dy_pct is not None:
        try:
            q_used = float(index_dy_pct) / 100.0
            source = "index_dividend_yield"
        except (TypeError, ValueError):
            q_used = FALLBACK_DIVIDEND_YIELD
            source = "static_fallback"
    else:
        q_used = FALLBACK_DIVIDEND_YIELD
        source = "static_fallback"

    return q_used, source, b, F


def classify_quote(bid, ask, mid) -> str:
    """ok vs wide_quote_low_liquidity, based on relative bid-ask spread."""
    if not bid or not ask or not mid:
        return "ok"  # spread check doesn't apply if we don't have both sides
    spread_pct = (ask - bid) / mid if mid else None
    if spread_pct is not None and spread_pct > WIDE_QUOTE_SPREAD_PCT:
        return "wide_quote_low_liquidity"
    return "ok"


# ---------------------------------------------------------------------------
# Put-call parity check (Phase 5, first of the data-quality checks)
#
# C - P == S*e^(-qT) - K*e^(-rT) is an exact identity on the model side --
# tests/test_greeks.py pins it on greeks.bs_price(). Applied to NSE's OBSERVED
# prices it holds only within a band, and the width of that band is the whole
# design problem: see parity_tolerance().
#
# Sign convention, load-bearing: deviation = observed - expected, so POSITIVE
# means calls are rich relative to puts. The same signed value is written to
# BOTH legs, un-negated, because the deviation is a property of the pair and
# not of either leg -- a PE-only query still reads "calls rich" correctly.
# ---------------------------------------------------------------------------

def _append_flag(existing, new) -> str:
    """
    Pipe-joins a data-quality flag onto whatever the row already carries, so a
    parity violation never destroys the wide-quote or no-IV verdict already
    there.

    "ok" is the IDENTITY element: appending to it yields the bare new flag,
    never "ok|parity_violation", so "ok" keeps meaning "nothing wrong". Order
    is fixed (existing flag first) so plain string equality still works.
    Parse the result with .split("|"), never by substring match.
    """
    if not existing or existing == "ok":
        return new
    parts = existing.split("|")
    if new in parts:
        return existing
    return "|".join(parts + [new])


def _parity_leg(row):
    """
    (price, half_spread) for one leg, or None if it cannot take part.

    The basis is the row's OWN price_source_for_iv -- the caller requires both
    legs to agree on it, which is what keeps a live mid from ever being
    compared against an hours-old trade, and is also what makes the basis
    recoverable downstream from either row alone (no self-join, no extra
    column).
    """
    source = row.get("price_source_for_iv")
    if source == "mid_price":
        price, bid, ask = row.get("mid_price"), row.get("bid_price"), row.get("ask_price")
        if price is None or bid is None or ask is None:
            return None
        # Clamped: a crossed quote must never SHRINK the tolerance.
        return price, max(0.0, (ask - bid) / 2.0)
    if source == "ltp":
        price = row.get("ltp")
        if not price:
            return None
        # No quotes exist here, so the spread term would vanish -- leaving the
        # TIGHTEST tolerance on the least reliable prices, which is backwards.
        # Substitute the wide-quote threshold instead: a strike with no
        # two-sided quote is at least as illiquid as that.
        return price, WIDE_QUOTE_SPREAD_PCT / 2.0 * price
    return None


def parity_tolerance(K, T, r, half_spread_ce, half_spread_pe) -> float:
    """
    The band inside which an observed C - P cannot be called a violation.

    Two terms plus a floor:

    - The quotable band (the two half-spreads). You cannot pin a price down
      more precisely than the spread you would have to cross, so this is what
      "demonstrably inconsistent" costs -- and it scales with liquidity for
      free, which is why the tolerance is not a flat point value.

    - Rate uncertainty. RISK_FREE_RATE is a static assumption and parity is
      sensitive to it through K*e^(-rT). Derived from the discount factor
      itself rather than bucketed by tenor, so it scales by strike and time
      automatically. This is the FULL width of the [r-delta, r+delta] band,
      algebraically 2*K*e^(-rT)*sinh(delta*T), so delta=50bp here carries the
      same allowance a one-sided 100bp bound would. That doubling is
      deliberate conservatism, NOT a missing factor of two -- halving it
      tightens the whole check by 2x. At K=25000, r=0.065: ~4.79 points at
      T=7/365, ~20.4 at T=30/365.

    Dividend-yield uncertainty is deliberately absent. The symmetric
    S*(e^(-(q-dq)T) - e^(-(q+dq)T)) term adds only ~1.2 points at dq=25bp and
    T=7/365 against the rate term's ~4.79, and q comes from NSE's PUBLISHED
    index dy rather than a static guess of ours. If weekly false positives
    ever cluster on high-dy days, that term is the drop-in.
    """
    rate_band = K * (math.exp(-(r - PARITY_RATE_DELTA) * T)
                     - math.exp(-(r + PARITY_RATE_DELTA) * T))
    return half_spread_ce + half_spread_pe + rate_band + PARITY_TICK_FLOOR


def flag_parity_violations(rows: list, symbol: str = "") -> dict:
    """
    Runs the put-call parity check over ONE cycle's rows, mutating them in
    place. Returns {expiry: {"checked", "violations", "max_abs_deviation"}}.

    MUST run after the whole per-leg loop has finished: parity needs both legs
    of a strike, and that loop sees exactly one leg at a time.

    Pairs are keyed on (expiry_date, strike), NEVER on strike alone -- since
    Phase 4 a single rows list carries two expiries at overlapping strikes, so
    keying on strike alone would pair a weekly CE against a monthly PE and
    flag essentially everything.

    parity_deviation is written on every CHECKED row, violating or not: the
    distribution of the clean ones is what lets the tolerance be re-tuned in a
    query later instead of by re-fetching a day. data_quality_flag is only
    touched on an actual violation.

    Pure: no config, no clock, no network. Every input is read off the rows
    themselves -- including risk_free_rate_used and dividend_yield_used rather
    than the module constants, so the check measures the chain against the
    discounting the pipeline actually applied to THAT row.
    """
    pairs = {}
    for row in rows:
        key = (row.get("expiry_date"), row.get("strike"))
        pairs.setdefault(key, {})[row.get("option_type")] = row

    stats = {}
    for (expiry, _strike), legs in pairs.items():
        ce, pe = legs.get("CE"), legs.get("PE")
        if ce is None or pe is None:
            continue
        # mid-vs-mid or ltp-vs-ltp, never one of each.
        if ce.get("price_source_for_iv") != pe.get("price_source_for_iv"):
            continue
        ce_leg, pe_leg = _parity_leg(ce), _parity_leg(pe)
        if ce_leg is None or pe_leg is None:
            continue

        # Both legs come from the same v3 response, so these agree; CE's copy
        # is taken arbitrarily. r/q are None on the rows that hit the
        # no_price_available guardrails (they are set only after those
        # continue), which is exactly what excludes those rows here.
        S = ce.get("underlying_value")
        K = ce.get("strike")
        T = ce.get("time_to_expiry_years")
        r = ce.get("risk_free_rate_used")
        q = ce.get("dividend_yield_used")
        if S is None or K is None or T is None or r is None or q is None or T <= 0:
            continue

        deviation = ((ce_leg[0] - pe_leg[0])
                     - (S * math.exp(-q * T) - K * math.exp(-r * T)))
        violated = abs(deviation) > parity_tolerance(
            K, T, r, ce_leg[1], pe_leg[1])

        for row in (ce, pe):
            row["parity_deviation"] = deviation
            if violated:
                row["data_quality_flag"] = _append_flag(
                    row.get("data_quality_flag"), PARITY_FLAG)

        s = stats.setdefault(expiry, {"checked": 0, "violations": 0,
                                      "max_abs_deviation": 0.0})
        s["checked"] += 1
        s["violations"] += int(violated)
        s["max_abs_deviation"] = max(s["max_abs_deviation"], abs(deviation))

    for expiry, s in sorted(stats.items()):
        nse_fetch.log(f"  [{symbol}] parity {expiry}: {s['checked']} pair(s) "
                      f"checked, {s['violations']} violation(s), "
                      f"max |dev| {s['max_abs_deviation']:.2f}")
        if (s["checked"] >= PARITY_MIN_SAMPLE
                and s["violations"] > PARITY_WARN_RATE * s["checked"]):
            gha_warning(
                f"[{symbol}] put-call parity violated on {s['violations']}/"
                f"{s['checked']} pair(s) for expiry {expiry} -- a rate that "
                f"high usually means something structural (wrong underlying "
                f"value, wrong expiry mapping), not a few stale strikes."
            )

    return stats


def _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_raw,
                     index_snapshot_raw, status):
    """
    One cycle's untouched NSE responses, as archived under vault/raw/.

    NOTE ON v3_raw's SHAPE -- the raw archive has two eras, and a
    rebuild-from-raw must tell them apart POSITIONALLY, not by date-parsing
    every key:
      - pre-Phase-4: a single v3 response, i.e. a dict with a "records" key.
      - Phase 4 on:  {expiry_string: v3_response, ...}, one entry per target
        expiry fetched that cycle (1 during expiry week, otherwise 2).
    The early-return paths below still pass a single response or None, since
    those are failures where no chain was kept at all.
    """
    return {
        "fetch_ts_utc": fetch_ts_utc.isoformat(),
        "fetch_ts_ist": fetch_ts_ist.isoformat(),
        "symbol": symbol,
        "responses": {
            "option_chain_bootstrap": bootstrap_raw,
            "option_chain_v3": v3_raw,
            "index_snapshot": index_snapshot_raw,
        },
        "fetch_status": status,
    }


def process_symbol(symbol: str, fetch_ts_utc: datetime, fetch_ts_ist: datetime,
                    index_snapshot_raw: dict):
    """
    Fetches and processes one symbol end to end. Returns (rows, snapshot).
    Raises only on truly unexpected errors the caller should log and move
    past -- most expected failure modes are handled internally and
    reflected in per-row flags / status, not exceptions.

    Two-step fetch (see nse_fetch.py's option-chain-v3 section for why):
      1. Bootstrap via fetch_option_chain() (the flat endpoint) -- gives a
         guaranteed-valid expiry string and futures/cost-of-carry data.
      2. Call option-chain-v3 with that expiry to discover the FULL expiry
         list, then fetch TWO target chains from it: the nearest upcoming
         expiry and the nearest monthly. Both land in the same rows list;
         expiry_date already tells them apart, so no new column is needed.
         The two picks coincide during expiry week, and either can coincide
         with the bootstrap expiry, so the chains are fetched through a small
         cache that turns every coincidence into a skipped call (0-2 extra v3
         calls per symbol). Unlike the flat endpoint, v3 carries actual
         bid/ask and NSE's own published IV.
    """
    status = {
        "option_chain": "ok", "futures": "ok", "lot_size": "static_fallback_only",
        "option_chain_v3": "ok",
        # Per-expiry breakdown: expiry -> {"legs": n, "status": ...}. With two
        # target expiries, the single option_chain_v3 status above can no
        # longer say WHICH chain arrived, and a half-fetched cycle would
        # otherwise be invisible in the raw archive.
        "expiries": {},
    }

    # --- Step 1: bootstrap (flat endpoint) -- futures + a valid expiry ---
    bootstrap_raw = nse_fetch.fetch_option_chain(symbol)
    bootstrap_entries = (bootstrap_raw or {}).get("data", [])

    if not bootstrap_entries:
        status["option_chain"] = "empty_response"
        gha_warning(f"[{symbol}] bootstrap option chain had no entries -- "
                    f"cannot discover a valid expiry, skipping this symbol this cycle.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, None, index_snapshot_raw, status)

    futures_by_expiry = nse_fetch.parse_futures_from_entries(bootstrap_entries)
    if not futures_by_expiry:
        status["futures"] = "no_futures_entries_in_response"
        gha_warning(f"[{symbol}] no futures entries found in the bootstrap response; "
                    f"dividend yield will use static_fallback for all rows this cycle.")

    bootstrap_expiry = next(
        (e.get("expiryDate") for e in bootstrap_entries if e.get("instrumentType", "").startswith("OPT")),
        None,
    )
    if not bootstrap_expiry:
        status["option_chain_v3"] = "no_bootstrap_expiry_found"
        gha_warning(f"[{symbol}] could not find any option expiry in the bootstrap "
                    f"response to seed option-chain-v3 -- skipping this symbol this cycle.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, None, index_snapshot_raw, status)

    # --- Step 2: option-chain-v3 -- discover all expiries, pick two targets ---
    v3_bootstrap = nse_fetch.fetch_option_chain_v3(symbol, bootstrap_expiry)
    all_expiries = v3_bootstrap.get("records", {}).get("expiryDates", [])

    if not all_expiries:
        status["option_chain_v3"] = "empty_response"
        gha_warning(f"[{symbol}] option-chain-v3 returned no expiry list even with a "
                    f"known-valid bootstrap expiry ({bootstrap_expiry}) -- skipping this "
                    f"symbol this cycle. Check whether NSE changed this endpoint again.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_bootstrap, index_snapshot_raw, status)

    as_of = fetch_ts_ist.date()
    weekly_expiry = nse_fetch.pick_nearest_weekly_expiry(all_expiries, as_of)
    monthly_expiry = nse_fetch.pick_nearest_monthly_expiry(all_expiries, as_of)

    # Ordered and de-duplicated: dict.fromkeys preserves weekly-then-monthly
    # order and collapses weekly == monthly (expiry week, when the front
    # contract IS the monthly) into ONE target. This is the dedupe that stops
    # the same chain being fetched, flattened and archived twice.
    target_expiries = [e for e in dict.fromkeys([weekly_expiry, monthly_expiry]) if e]

    if not target_expiries:
        status["option_chain_v3"] = "no_parseable_expiry"
        gha_warning(f"[{symbol}] none of the {len(all_expiries)} expiry string(s) NSE "
                    f"returned could be parsed as DD-Mon-YYYY -- skipping this symbol "
                    f"this cycle. Check whether NSE changed the expiry format.")
        return [], _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_bootstrap, index_snapshot_raw, status)

    # Seed the cache with the response already in hand, so a target equal to
    # the bootstrap expiry costs no call at all. Keyed by the expiry STRING
    # exactly as it would be requested: if the flat endpoint's expiryDate ever
    # stopped matching v3's expiryDates format, the worst outcome is ONE
    # redundant call -- never a duplicated or a missing chain, because the rows
    # below are built from target_expiries, never from this dict's keys.
    v3_by_expiry = {bootstrap_expiry: v3_bootstrap}
    extra_calls = [e for e in target_expiries if e not in v3_by_expiry]

    nse_fetch.log(
        f"  [{symbol}] expiries this cycle: weekly={weekly_expiry} "
        f"monthly={monthly_expiry} -> fetching {len(target_expiries)} chain(s): "
        f"{', '.join(target_expiries)} (bootstrap={bootstrap_expiry}, "
        f"{len(extra_calls)} extra v3 call(s))"
    )

    for expiry in extra_calls:
        try:
            v3_by_expiry[expiry] = nse_fetch.fetch_option_chain_v3(symbol, expiry)
        except Exception as exc:  # noqa: BLE001
            # Per-expiry isolation, mirroring main()'s per-symbol isolation: a
            # monthly that fails after all its retries must not cost this
            # symbol the weekly chain already sitting in memory.
            status["expiries"][expiry] = {
                "legs": 0, "status": "fetch_failed", "error": str(exc),
            }
            gha_warning(f"[{symbol}] option-chain-v3 fetch failed for expiry {expiry} "
                        f"after retries ({exc}); continuing with the other expiry.")

    legs_by_expiry = {}
    for expiry in target_expiries:
        if expiry in status["expiries"]:  # already recorded above as a failed fetch
            legs_by_expiry[expiry] = []
            continue
        legs = nse_fetch.flatten_v3_entries(v3_by_expiry.get(expiry))
        legs_by_expiry[expiry] = legs
        status["expiries"][expiry] = {
            "legs": len(legs), "status": "ok" if legs else "empty",
        }
        if not legs:
            gha_warning(f"[{symbol}] option-chain-v3 returned no CE/PE legs for "
                        f"expiry {expiry}.")

    # Both expiries' legs, weekly first, in ONE list -- the row loop below is
    # unchanged by this: it already reads expiry_date and T per leg.
    flat_legs = [leg for expiry in target_expiries for leg in legs_by_expiry[expiry]]

    nse_fetch.log(
        f"  [{symbol}] legs fetched: "
        + ", ".join(f"{e}={len(legs_by_expiry[e])}" for e in target_expiries)
        + f" ({len(flat_legs)} total)"
    )

    # Roll the per-expiry outcomes up into the single status the pre-Phase-4
    # raw archive already carried, keeping "empty_target_expiry_response"
    # meaning exactly what it meant before: nothing usable came back at all.
    ok_expiries = [e for e, s in status["expiries"].items() if s["status"] == "ok"]
    if not ok_expiries:
        status["option_chain_v3"] = "empty_target_expiry_response"
        gha_warning(f"[{symbol}] no target expiry produced any CE/PE legs -- this "
                    f"symbol will fail its freshness check this cycle.")
    elif len(ok_expiries) < len(target_expiries):
        status["option_chain_v3"] = "partial_expiry_failure"

    lot_size = config_loader.lot_size(symbol)  # no live source exists anymore

    idx_ohlc = nse_fetch.parse_index_snapshot(index_snapshot_raw, symbol) if index_snapshot_raw else {}
    india_vix_row = nse_fetch.parse_india_vix(index_snapshot_raw) if index_snapshot_raw else {}
    india_vix = india_vix_row.get("last")

    rows = []
    for leg in flat_legs:
        opt_type = leg.get("option_type")
        expiry_date = leg.get("expiryDates")  # consistent DD-Mon-YYYY format
        raw_strike = leg.get("strikePrice")
        try:
            strike = float(raw_strike)
        except (TypeError, ValueError):
            continue

        T = years_to_expiry(expiry_date, fetch_ts_ist) if expiry_date else None
        entry_underlying = leg.get("underlyingValue")

        bid = leg.get("buyPrice1") or 0.0
        bid_qty = leg.get("buyQuantity1")
        ask = leg.get("sellPrice1") or 0.0
        ask_qty = leg.get("sellQuantity1")
        mid = (bid + ask) / 2 if (bid and ask) else None
        ltp = leg.get("lastPrice") or 0.0
        nse_iv = leg.get("impliedVolatility")
        pchange = leg.get("pChange")

        row = {c: None for c in vault_io.MAIN_COLUMNS}
        row.update({
            "fetch_ts_utc": fetch_ts_utc.isoformat(),
            "fetch_ts_ist": fetch_ts_ist.isoformat(),
            "symbol": symbol,
            "poller_id": POLLER_ID,
            "expiry_date": expiry_date,
            "strike": strike,
            "option_type": opt_type,
            "underlying_value": entry_underlying,
            "bid_price": bid or None,
            "bid_qty": bid_qty,
            "ask_price": ask or None,
            "ask_qty": ask_qty,
            "total_buy_quantity": leg.get("totalBuyQuantity"),
            "total_sell_quantity": leg.get("totalSellQuantity"),
            "ltp": ltp,
            "mid_price": mid,
            "open_interest": leg.get("openInterest"),
            "change_in_oi": leg.get("changeinOpenInterest"),
            "total_traded_volume": leg.get("totalTradedVolume"),
            "pchange_vs_prev_close": pchange,
            "nse_iv": nse_iv,
            "time_to_expiry_years": T,
            "india_vix": india_vix,
            "lot_size": lot_size,
            "underlying_day_open": idx_ohlc.get("open"),
            "underlying_day_high": idx_ohlc.get("high"),
            "underlying_day_low": idx_ohlc.get("low"),
            "underlying_prev_close": idx_ohlc.get("prev_close"),
        })

        # Real bid/ask are back -- prefer mid-price over LTP again, exactly
        # like the original design intended. Used here purely for the
        # below-intrinsic-value guardrail now (see below), since we no
        # longer solve our own IV from it -- Greeks use nse_iv directly.
        price_for_iv = mid if mid else (ltp if ltp else None)
        row["price_source_for_iv"] = "mid_price" if mid else ("ltp" if ltp else "none")

        if not price_for_iv or not entry_underlying or not strike or not T or T <= 0:
            row["data_quality_flag"] = "no_price_available"
            rows.append(row)
            continue

        # Guardrail, preserved even without our own solver: a price below
        # intrinsic value is a strong sign of a stale/bad quote, regardless
        # of what IV NSE has published for it.
        intrinsic = max(entry_underlying - strike, 0.0) if opt_type == "CE" else max(strike - entry_underlying, 0.0)
        if price_for_iv < intrinsic - 1e-6:
            row["data_quality_flag"] = "no_price_available"
            rows.append(row)
            continue

        # futures_by_expiry is built from the bootstrap response's FUT* legs,
        # which NSE lists for MONTHLY expiries only. A nearest-weekly row
        # therefore gets F=None (and so implied_cost_of_carry=None) except
        # during expiry week, when the weekly IS the monthly. That is expected,
        # not a gap to "fix": both are informational columns, and
        # dividend_yield_used comes from the index's published dy, not from F.
        q_used, carry_source, b, F = get_dividend_yield_and_carry(
            futures_by_expiry, expiry_date, entry_underlying, T, idx_ohlc.get("dy"),
        )
        row["futures_price"] = F
        row["implied_cost_of_carry"] = b
        row["dividend_yield_used"] = q_used
        row["dividend_yield_source"] = carry_source
        row["risk_free_rate_used"] = RISK_FREE_RATE

        # Greeks now use NSE's own published IV directly -- no in-house
        # solver in the loop. nse_iv arrives as a percentage (e.g. 8.8
        # means 8.8%), so it's converted to decimal here.
        sigma = None
        if nse_iv is not None:
            try:
                candidate = float(nse_iv) / 100.0
                if candidate > 0:
                    sigma = candidate
            except (TypeError, ValueError):
                sigma = None

        if sigma is None:
            row["data_quality_flag"] = "no_nse_iv"
            rows.append(row)
            continue

        g = greeks.compute_all_greeks(
            entry_underlying, strike, T, RISK_FREE_RATE, q_used, sigma, opt_type,
        )
        row.update(g)
        row["data_quality_flag"] = classify_quote(bid or None, ask or None, mid)

        rows.append(row)

    # Put-call parity, over the COMPLETED rows list. It cannot run inside the
    # loop above: a parity check needs both legs of a strike, and that loop
    # sees one leg at a time. Running it here also makes it work unchanged for
    # the two-expiry case, since rows already holds both.
    #
    # A new top-level status key rather than extra keys inside
    # status["expiries"] -- tests/test_process_symbol_expiries.py asserts that
    # dict's exact shape, and the per-expiry leg counts mean something
    # different from the per-expiry parity verdict.
    status["parity"] = flag_parity_violations(rows, symbol)

    # Archive one raw response per target expiry, keyed by that expiry. Only
    # the targets: a bootstrap response for an expiry we did not keep rows for
    # contributes nothing to MAIN, exactly as before. Keying by expiry (rather
    # than storing a single response as pre-Phase-4 cycles did) lets a replay
    # take .values() without knowing how many chains a cycle fetched -- see the
    # vault-schema skill for the pre/post-Phase-4 shape rule.
    v3_for_snapshot = {e: v3_by_expiry[e] for e in target_expiries if e in v3_by_expiry}
    snapshot = _build_snapshot(symbol, fetch_ts_utc, fetch_ts_ist, bootstrap_raw, v3_for_snapshot, index_snapshot_raw, status)
    return rows, snapshot


def main():
    run_start = time.monotonic()
    fetch_ts_utc = datetime.now(timezone.utc)
    fetch_ts_ist = fetch_ts_utc.astimezone(IST)
    today = fetch_ts_ist.date()

    if not holidays.is_trading_day(today):
        print(f"{today.isoformat()} is not a trading day (weekend or holiday). "
              f"Skipping run cleanly -- no files touched.")
        sys.exit(0)

    print(f"Fetch cycle started at {fetch_ts_ist.isoformat()} (IST)")

    # One call, reused across all three symbols (day OHLC + India VIX).
    # If this fails, we don't abort the whole run -- OHLC/VIX columns will
    # just be null for this cycle, everything else still gets fetched.
    index_snapshot_raw = None
    try:
        index_snapshot_raw = nse_fetch.fetch_index_snapshot_raw()
    except Exception as exc:  # noqa: BLE001
        gha_warning(f"Index snapshot (OHLC + India VIX) fetch failed after retries "
                    f"({exc}); those columns will be null for this cycle.")

    symbols = config_loader.symbol_names()

    # Warm the shared HTTP session ONCE, explicitly, while still single-
    # threaded. The lazy per-call init races if several threads try to create
    # the session at the same instant (see nse_fetch._get_session), so we force
    # it here before spawning any workers. Calling symbol_names() above also
    # pre-populated config_loader's cache in this thread, so the worker threads
    # only ever READ config, never race to load it.
    nse_fetch.warmup_session()

    # Fetch + process every symbol CONCURRENTLY. The symbols are fully
    # independent and the work is almost entirely NSE network I/O, so a small
    # thread pool turns a cycle from sum-of-symbols into roughly max-of-symbols.
    # Each symbol still fails independently: an exception from one symbol's
    # process_symbol() is caught below and never blocks the others (mirrors the
    # old per-symbol try/except from the sequential loop).
    for symbol in symbols:
        print(f"Processing {symbol}...")

    results = {}  # symbol -> (rows, snapshot), or None if that symbol failed
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(symbols))) as executor:
        futures = {
            symbol: executor.submit(
                process_symbol, symbol, fetch_ts_utc, fetch_ts_ist, index_snapshot_raw
            )
            for symbol in symbols
        }
        for symbol in symbols:
            try:
                results[symbol] = futures[symbol].result()
            except Exception as exc:  # noqa: BLE001
                gha_error(f"[{symbol}] unexpected failure, skipping this symbol this cycle: {exc}")
                results[symbol] = None

    # All worker threads have joined here. Do the freshness check + disk writes
    # in the main thread, in config order: this keeps vault_io's (unlocked) log
    # lines from interleaving with the concurrent fetch logs above, keeps write
    # ordering deterministic, and serializes disk access. Writes are fast local
    # I/O -- the latency win was in parallelizing the network fetches, not here.
    any_symbol_succeeded = False
    for symbol in symbols:
        result = results.get(symbol)
        if result is None:
            continue
        rows, snapshot = result

        underlying_value = rows[0]["underlying_value"] if rows else None
        if not vault_io.freshness_ok(symbol, underlying_value, rows):
            gha_warning(f"[{symbol}] failed freshness check -- not writing MAIN rows "
                        f"for this symbol this cycle (raw archive still saved for audit).")
            # Still archive the raw response even on freshness failure --
            # useful for debugging why NSE returned something unusable.
            try:
                vault_io.append_raw_snapshot(symbol, today, snapshot)
            except Exception as exc:  # noqa: BLE001
                gha_warning(f"[{symbol}] could not write raw archive either: {exc}")
            continue

        try:
            vault_io.append_raw_snapshot(symbol, today, snapshot)
        except Exception as exc:  # noqa: BLE001
            gha_warning(f"[{symbol}] raw archive write failed: {exc} (MAIN rows still kept)")

        # Each symbol gets its OWN MAIN shard, one per fetch cycle, under
        # vault/tables/<SYMBOL>/<YYYY-MM-DD>/ -- per symbol, not accumulated
        # and merged across symbols into one shared file. fetch_ts_ist (not
        # `today`) is passed because the shard name carries the time too, which
        # is what makes a re-run of THIS cycle overwrite its own shard instead
        # of adding a duplicate.
        try:
            shard = vault_io.write_main_shard(symbol, fetch_ts_ist, rows)
        except Exception as exc:  # noqa: BLE001
            # Per-symbol isolation, matching what the concurrent fetch loop and
            # the raw-archive write above already have. Writes are serialized here in config order, so without this an ENOSPC or a
            # permission error on the FIRST symbol's shard would propagate out
            # of the loop and cost every LATER symbol its write too -- symbols
            # that fetched fine and whose rows are sitting in memory, ready.
            # The raw snapshot for this symbol is already on disk by this
            # point, so the cycle stays reconstructible from raw.
            gha_error(f"[{symbol}] MAIN shard write failed: {exc} -- no MAIN rows for "
                      f"this symbol this cycle (raw archive still saved).")
            continue

        # Gated on the actual return, not on "we called the writer": if every
        # row failed the schema's identity requirements there is no shard on
        # disk, and this symbol did not succeed. write_main_shard() already
        # logs the path and the true written row count, which can be lower
        # than len(rows) when rows are dropped, so there's no second print here.
        if shard:
            any_symbol_succeeded = True

    if not any_symbol_succeeded:
        gha_error("No symbol produced usable data this cycle. Check NSE endpoint "
                  "health and the warnings/errors above.")
        # Exit 0 anyway -- a single bad cycle shouldn't fail the whole
        # scheduled workflow (retries + the next cycle will likely recover).
        # The ::error:: annotation above still makes this visible in the
        # Actions run summary.

    total_elapsed = time.monotonic() - run_start
    print(f"Fetch cycle finished in {total_elapsed:.1f}s total.")
    sys.exit(0)


if __name__ == "__main__":
    main()
