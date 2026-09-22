"""
holidays.py

Simple, manually-maintained NSE trading holiday list. No reliable free API
for the NSE trading calendar was found, so this is edited by hand each
year -- put a reminder on your calendar to update NSE_HOLIDAYS for 2027
once NSE publishes next year's list (usually released a few months before
year end).

Unexpected/one-off closures NOT on this list (e.g. an ad-hoc circuit
closure) are deliberately NOT handled here -- the freshness check in
vault_io.py already skips writing on a day where NSE returns no usable
data, so this list only needs to cover PLANNED closures known in advance.
"""

from datetime import date

# Format: date(year, month, day)
NSE_HOLIDAYS_2026 = {
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali - Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}

# Add more NSE_HOLIDAYS_<year> sets here as years pass, then include them
# in ALL_HOLIDAYS below.
ALL_HOLIDAYS = set() | NSE_HOLIDAYS_2026


def is_trading_day(d: date) -> bool:
    """False for Saturdays, Sundays, and any date in ALL_HOLIDAYS."""
    if d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    if d in ALL_HOLIDAYS:
        return False
    return True
