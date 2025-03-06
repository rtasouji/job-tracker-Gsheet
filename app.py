import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import sys
import os
import pandas as pd

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
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(CREDS_JSON), SCOPE)
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

if __name__ == "__main__":
    campaign_name = "default" if len(sys.argv) < 2 else sys.argv[1]
    check_db(campaign_name)
