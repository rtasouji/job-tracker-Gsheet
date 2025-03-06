import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import sys
import pandas as pd
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("job-tracker")

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_JSON = os.getenv("GOOGLE_SHEETS_CREDS")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

def check_db(campaign_name="default"):
    try:
        logger.info(f"GOOGLE_SHEETS_CREDS length: {len(CREDS_JSON) if CREDS_JSON else 0}")
        creds_dict = json.loads(CREDS_JSON)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        worksheet = client.open_by_key(SPREADSHEET_ID).worksheet("share_of_voice")
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        today = datetime.date.today()
        today_records = df[(df["date"] == str(today)) & (df["campaign_name"] == campaign_name)]
        count = len(today_records)
        
        logger.info(f"Records found for today for campaign '{campaign_name}': {count}")
        if count == 0:
            logger.warning(f"⚠️ WARNING: No records were stored for today for campaign '{campaign_name}'!")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in GOOGLE_SHEETS_CREDS: {e}")
        raise
    except Exception as e:
        logger.error(f"Error checking Google Sheets for campaign '{campaign_name}': {e}")
        raise

if __name__ == "__main__":
    campaign_name = "default" if len(sys.argv) < 2 else sys.argv[1]
    check_db(campaign_name)
