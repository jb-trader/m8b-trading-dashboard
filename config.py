"""
M8B Trading Dashboard Configuration
Cloud deployment version
"""

GOOGLE_DRIVE_FILE_ID = "1-N9DUSIxm0zSE9YEor0G9F1IoUwbSQG9"  

def get_data_url():
    """Generate direct download URL for Google Drive"""
    return f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# Version info
VERSION = "2.0.0"
DEPLOYMENT = "Cloud"
LAST_DATA_UPDATE = "Updates daily after market close"

# Exclusion dates
FOMC_DATES = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18"
]

EARNINGS_DATES = [
    "2024-01-30", "2024-02-01", "2024-02-21", "2024-03-07",
    "2024-04-24", "2024-04-25", "2024-04-30", "2024-05-02",
    "2024-05-22", "2024-06-12", "2024-07-30", "2024-07-31",
    "2024-08-01", "2024-08-28", "2024-09-05", "2024-10-30",
    "2024-10-31", "2024-11-20", "2024-12-12", "2025-01-29",
    "2025-01-30", "2025-02-06", "2025-02-26", "2025-03-06"
]