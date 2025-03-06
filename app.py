import streamlit as st
import requests
import pandas as pd
import tldextract
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from collections import defaultdict
import datetime
import os
import plotly.graph_objects as go
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("job-tracker")

# Google Sheets setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_JSON = os.getenv("GOOGLE_SHEETS_CREDS")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

# Debug environment variables
logger.info(f"GOOGLE_SHEETS_CREDS length: {len(CREDS_JSON) if CREDS_JSON else 0}")
logger.info(f"SPREADSHEET_ID: {SPREADSHEET_ID}")

if not CREDS_JSON or not SPREADSHEET_ID:
    raise ValueError("❌ ERROR: GOOGLE_SHEETS_CREDS or SPREADSHEET_ID environment variable is not set!")

def get_sheets_client():
    try:
        creds_dict = json.loads(CREDS_JSON)  # Parse the JSON string into a Python dict
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        return gspread.authorize(creds)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in GOOGLE_SHEETS_CREDS: {e}")
        raise ValueError(f"❌ ERROR: GOOGLE_SHEETS_CREDS is not valid JSON: {CREDS_JSON}")
    except Exception as e:
        logger.error(f"Error creating Sheets client: {e}")
        raise

# Rest of your functions (unchanged unless specified)
def get_worksheet(client, sheet_name):
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    try:
        return spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=10)

def initialize_sheets():
    client = get_sheets_client()
    get_worksheet(client, "share_of_voice")
    get_worksheet(client, "campaigns")
    logger.info("Google Sheets initialized")

initialize_sheets()

# ... (rest of your original app.py code here, like load_jobs, get_google_jobs_results, etc.)

# Example check_db function (if this is what you meant)
def check_db(campaign_name="default"):
    client = get_sheets_client()
    worksheet = get_worksheet(client, "share_of_voice")
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    
    today = datetime.date.today()
    today_records = df[(df["date"] == str(today)) & (df["campaign_name"] == campaign_name)]
    count = len(today_records)
    
    logger.info(f"Records found for today for campaign '{campaign_name}': {count}")
    if count == 0:
        logger.warning(f"⚠️ WARNING: No records were stored for today for campaign '{campaign_name}'!")

# Streamlit UI and other logic (unchanged)
if __name__ == "__main__":
    # If running via command line with 'github' argument
    if len(sys.argv) > 1 and sys.argv[1] == "github":
        campaign_name = sys.argv[2] if len(sys.argv) > 2 else "default"
        print(f"🚀 Running automated fetch & store process for campaign: {campaign_name}")
        sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov(campaign_name)
        save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, campaign_name)
        compute_and_store_total_data()
        print("✅ Data stored successfully!")
    else:
        # Normal Streamlit execution
        st.title("Google for Jobs Visibility Tracker")
        # ... (rest of your Streamlit code)
