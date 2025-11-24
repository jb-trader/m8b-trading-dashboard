from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

def current_data_version_at_430_et() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    cutoff = time(16, 30)  # 4:30 pm ET
    data_day = now_et.date() if now_et.time() >= cutoff else (now_et - timedelta(days=1)).date()
    return data_day.strftime("%Y-%m-%d")

GOOGLE_DRIVE_FILE_ID = "1-N9DUSIxm0zSE9YEor0G9F1IoUwbSQG9"

def get_data_version() -> str:
    """Compute version each call so it flips at 4:30pm ET without reload."""
    return current_data_version_at_430_et()

def get_data_url() -> str:
    """Include version query to bust HTTP caches."""
    return f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&v={get_data_version()}"

# Meta
VERSION = "2.0.0"
DEPLOYMENT = "Cloud"
LAST_DATA_UPDATE = "Updates daily after market close"

# FOMC_DATES and EARNINGS_DATES ... (your lists are fine)


# Exclusion dates
#  use this source:   https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"
]

# NVDA, MSFT, AAPL, AMZN, META, AVGO 

EARNINGS_DATES = [
 
    "2024-01-30", "2024-02-01", "2024-02-21", "2024-04-24", "2024-04-25",
    "2024-04-30", "2024-05-02", "2024-05-22", "2024-07-23", "2024-07-30",

    "2024-07-31", "2024-08-01", "2024-08-28", "2024-10-29", "2024-10-30",
    "2024-10-31", "2024-11-20", "2025-01-29", "2025-01-30", "2025-02-04",

    "2025-02-06", "2025-02-26", "2025-04-24", "2025-04-30", "2025-05-01",
    "2025-05-28", "2025-07-23", "2025-07-30", "2025-07-31", "2025-08-27",

    "2025-11-19", "2025-10-29", "2025-10-30", "2025-12-11"
]