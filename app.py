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
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs.txt"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("job-tracker")

# Google Sheets setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_JSON_RAW = os.getenv("GOOGLE_SHEETS_CREDS")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
SERP_API_KEY = os.getenv("SERP_API_KEY")

if CREDS_JSON_RAW:
    try:
        CREDS_JSON = json.loads(CREDS_JSON_RAW)
        if isinstance(CREDS_JSON, str):
            logger.info("Detected double-encoded GOOGLE_SHEETS_CREDS, parsing again")
            CREDS_JSON = json.loads(CREDS_JSON)
        logger.info(f"Successfully parsed GOOGLE_SHEETS_CREDS into dict with keys: {list(CREDS_JSON.keys())}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GOOGLE_SHEETS_CREDS: {e}")
        CREDS_JSON = None
else:
    CREDS_JSON = None

if not CREDS_JSON or not SPREADSHEET_ID or not SERP_API_KEY:
    missing_vars = [var for var, val in [("GOOGLE_SHEETS_CREDS", CREDS_JSON), ("SPREADSHEET_ID", SPREADSHEET_ID), ("SERP_API_KEY", SERP_API_KEY)] if not val]
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Global cache for worksheet data
worksheet_cache = {}

# Cache for historical data
historical_data_cache = {}

# Rate limiting
last_request_time = 0

# Hardcoded admin credentials (you can change these or load from environment variables)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "securepassword123")

def rate_limit():
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < 1:  # Ensure at least 1 second between requests
        time.sleep(1 - time_since_last)
    last_request_time = time.time()

def get_sheets_client():
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(CREDS_JSON, SCOPE)
        logger.info("Sheets client created successfully")
        return gspread.authorize(creds)
    except Exception as e:
        logger.error(f"Error creating Sheets client: {e}")
        raise

def get_worksheet(client, sheet_name):
    try:
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            logger.info(f"Accessed existing worksheet: {sheet_name}")
            return worksheet
        except gspread.WorksheetNotFound:
            logger.info(f"Worksheet {sheet_name} not found, creating it")
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=10)
            worksheet.append_row(["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
            return worksheet
    except Exception as e:
        logger.error(f"Error accessing worksheet '{sheet_name}': {e}")
        raise

def initialize_sheets():
    try:
        client = get_sheets_client()
        # Cache the worksheets
        worksheet_cache["campaigns"] = get_worksheet(client, "campaigns")
        worksheet_cache["share_of_voice"] = get_worksheet(client, "share_of_voice")
        logger.info("Google Sheets initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Google Sheets: {e}")
        raise

def get_campaigns_by_country(country):
    try:
        worksheet = worksheet_cache["campaigns"]
        rate_limit()
        data = worksheet.get_all_records()
        country_campaigns = [row["campaign_name"] for row in data if row.get("country", "US").upper() == country.upper()]
        logger.info(f"Found {len(country_campaigns)} campaigns for country '{country}': {country_campaigns}")
        return country_campaigns
    except Exception as e:
        logger.error(f"Error fetching campaigns for country '{country}': {e}")
        return []

def fetch_campaigns_by_country(country):
    try:
        country_campaigns = get_campaigns_by_country(country)
        if not country_campaigns:
            logger.warning(f"No campaigns found for country '{country}' to process")
            return False, []

        successful_campaigns = []
        total_campaigns = len(country_campaigns)
        progress_bar = st.progress(0)
        for i, campaign_name in enumerate(country_campaigns):
            logger.info(f"Processing campaign: {campaign_name} for country '{country}'")
            try:
                sov_data, appearances, avg_v_rank, avg_h_rank, single_link, _ = compute_sov(campaign_name)
                save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, single_link, campaign_name, country)
                successful_campaigns.append(campaign_name)
                logger.info(f"Successfully processed campaign: {campaign_name}")
            except Exception as e:
                logger.error(f"Failed to process campaign '{campaign_name}': {e}")
                continue
            progress = (i + 1) / total_campaigns
            progress_bar.progress(progress)

        compute_and_store_total_data()
        logger.info(f"Processed {len(successful_campaigns)}/{len(country_campaigns)} campaigns for country '{country}' successfully")
        return True, successful_campaigns
    except Exception as e:
        logger.error(f"Error in fetch_campaigns_by_country for country '{country}': {e}")
        return False, []

def load_jobs(campaign_name):
    try:
        worksheet = worksheet_cache["campaigns"]
        data = worksheet.get_all_records()
        campaign_data = next((row for row in data if row["campaign_name"] == campaign_name), None)
        
        if not campaign_data:
            logger.warning(f"Campaign '{campaign_name}' not found")
            return []
        
        job_titles = json.loads(campaign_data["job_titles"]) if campaign_data["job_titles"] else []
        locations = json.loads(campaign_data["locations"]) if campaign_data["locations"] else []
        country = campaign_data.get("country", "US")
        
        if len(job_titles) != len(locations):
            logger.error(f"Mismatch between job titles ({len(job_titles)}) and locations ({len(locations)}) for campaign '{campaign_name}'")
            return []
        
        logger.info(f"Loaded {len(job_titles)} job queries for campaign '{campaign_name}' in country '{country}'")
        return [{"job_title": title, "location": loc, "country": country} for title, loc in zip(job_titles, locations)]
    except Exception as e:
        logger.error(f"Error in load_jobs for campaign '{campaign_name}': {e}")
        return []

def get_google_jobs_results(query, location):
    try:
        logger.info(f"Fetching results for query: '{query}' in location: '{location}'")
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_jobs",
            "q": query,
            "location": location if location else "United States",
            "hl": "en",
            "api_key": SERP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        logger.info(f"SerpAPI response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs_results", [])
        logger.info(f"Fetched {len(jobs)} jobs for query: '{query}'")
        return jobs
    except requests.Timeout:
        logger.error(f"Timeout fetching SerpAPI for '{query}'")
        return []
    except requests.RequestException as e:
        logger.error(f"SerpAPI request failed: {e}")
        return []

def extract_domain(url):
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from URL '{url}': {e}")
        return ""

def compute_sov(campaign_name):
    logger.info(f"Starting compute_sov for campaign '{campaign_name}'")
    domain_sov = defaultdict(float)
    domain_appearances = defaultdict(int)
    domain_v_rank = defaultdict(list)
    domain_h_rank = defaultdict(list)
    domain_single_link = defaultdict(int)
    total_sov = 0

    jobs_data = load_jobs(campaign_name)
    if not jobs_data:
        logger.warning(f"No job data for campaign '{campaign_name}'")
        return {}, {}, {}, {}, {}, "US"

    country = jobs_data[0]["country"] if jobs_data else "US"
    for job_query in jobs_data:
        job_title = job_query["job_title"]
        location = job_query["location"]
        jobs = get_google_jobs_results(job_title, location)
        for job_rank, job in enumerate(jobs, start=1):
            apply_options = job.get("apply_options", [])
            V = 1 / job_rank
            if len(apply_options) == 1 and "link" in apply_options[0]:
                domain = extract_domain(apply_options[0]["link"])
                domain_single_link[domain] += 1
                domain_sov[domain] += V
                domain_appearances[domain] += 1
                domain_v_rank[domain].append(job_rank)
                domain_h_rank[domain].append(1)
                total_sov += V
            else:
                for link_order, option in enumerate(apply_options, start=1):
                    if "link" in option:
                        domain = extract_domain(option["link"])
                        H = 1 / link_order
                        weight = V * H
                        domain_sov[domain] += weight
                        domain_appearances[domain] += 1
                        domain_v_rank[domain].append(job_rank)
                        domain_h_rank[domain].append(link_order)
                        total_sov += weight
        time.sleep(1)

    if total_sov > 0:
        domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}
    else:
        logger.warning(f"Total SoV is 0 for '{campaign_name}'")

    domain_avg_v_rank = {domain: round(sum(vr) / len(vr), 2) for domain, vr in domain_v_rank.items() if vr}
    domain_avg_h_rank = {domain: round(sum(hr) / len(hr), 2) for domain, hr in domain_h_rank.items() if hr}
    
    logger.info(f"Computed SoV for '{campaign_name}': {len(domain_sov)} domains, {sum(domain_single_link.values())} single-link appearances")
    return domain_sov, domain_appearances, domain_avg_v_rank, domain_avg_h_rank, domain_single_link, country

def save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, single_link, campaign_name, country):
    if not sov_data:
        logger.warning(f"No SoV data to save for campaign '{campaign_name}'")
        return
    
    try:
        worksheet = worksheet_cache["share_of_voice"]
        today = datetime.date.today().isoformat()
        logger.info(f"Saving data for campaign '{campaign_name}' on {today}")
        
        rate_limit()
        all_data = worksheet.get_all_records()
        df = pd.DataFrame(all_data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
        logger.info(f"Existing data: {len(df)} rows")
        
        if not df.empty:
            df_to_keep = df[(df["campaign_name"] != campaign_name) | (df["date"] != today)]
            logger.info(f"Keeping {len(df_to_keep)} rows after filtering")
        else:
            df_to_keep = pd.DataFrame(columns=df.columns)
            logger.info("No existing data to keep")
        
        new_rows = []
        for domain in sov_data:
            sov = sov_data[domain] if pd.notna(sov_data[domain]) else 0
            avg_v = avg_v_rank.get(domain, 0) if pd.notna(avg_v_rank.get(domain, 0)) else 0
            avg_h = avg_h_rank.get(domain, 0) if pd.notna(avg_h_rank.get(domain, 0)) else 0
            new_rows.append([domain, today, round(float(sov), 2), appearances[domain], 
                            float(avg_v), float(avg_h), campaign_name, country, single_link.get(domain, 0)])
        logger.info(f"New rows to add: {len(new_rows)}")
        
        updated_data = df_to_keep.values.tolist() if not df_to_keep.empty else []
        updated_data.extend(new_rows)
        logger.info(f"Total rows to write: {len(updated_data)}")
        
        if updated_data:
            rate_limit()
            worksheet.clear()
            worksheet.append_row(["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
            worksheet.append_rows(updated_data)
            logger.info(f"Replaced {len(new_rows)} rows for '{campaign_name}' on {today}")
        else:
            logger.warning(f"No data to write for '{campaign_name}' on {today}, sheet unchanged")
    except Exception as e:
        logger.error(f"Error saving to Google Sheets for campaign '{campaign_name}': {e}")
        raise

def get_historical_data(start_date, end_date, campaign_name):
    try:
        worksheet = worksheet_cache["share_of_voice"]
        rate_limit()
        data = worksheet.get_all_records()
        df = pd.DataFrame(data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
        if "single_link" not in df.columns:
            df["single_link"] = 0
        
        df = df[df["campaign_name"] == campaign_name]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        
        if df.empty:
            logger.info(f"No historical data for campaign '{campaign_name}' in range {start_date} to {end_date}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_agg = df.groupby(["domain", "date"], as_index=False).agg({
            "sov": "mean",
            "appearances": "sum",
            "avg_v_rank": "mean",
            "avg_h_rank": "mean",
            "single_link": "sum"
        })
        
        df_sov = df_agg.pivot(index="domain", columns="date", values="sov").fillna(0)
        df_metrics = df_agg.pivot(index="domain", columns="date", values=["appearances", "avg_v_rank", "avg_h_rank", "single_link"]).fillna(0)
        df_metrics = df_metrics.swaplevel(axis=1).sort_index(axis=1)
        df_appearances = df_agg.pivot(index="domain", columns="date", values="appearances").fillna(0)

        if not df_sov.empty:
            most_recent_date = df_sov.columns[-1]
            df_sov = df_sov.sort_values(by=most_recent_date, ascending=False)
        
        return df_sov, df_metrics, df_appearances
    except Exception as e:
        logger.error(f"Error in get_historical_data for '{campaign_name}': {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def get_total_historical_data(start_date, end_date, country):
    cache_key = f"{country}_{start_date}_{end_date}"
    if cache_key in historical_data_cache:
        logger.info(f"Using cached historical data for {cache_key}")
        return historical_data_cache[cache_key]

    try:
        worksheet = worksheet_cache["share_of_voice"]
        rate_limit()
        data = worksheet.get_all_records()
        df = pd.DataFrame(data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
        
        df = df[(df["campaign_name"] == f"Total - {country}") & (df["country"] == country)]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        
        if df.empty:
            logger.info(f"No 'Total - {country}' historical data in range {start_date} to {end_date}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_agg = df[["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "single_link"]]
        
        df_sov = df_agg.pivot(index="domain", columns="date", values="sov").fillna(0)
        df_metrics = df_agg.pivot(index="domain", columns="date", values=["appearances", "avg_v_rank", "avg_h_rank", "single_link"]).fillna(0)
        df_metrics = df_metrics.swaplevel(axis=1).sort_index(axis=1)
        df_appearances = df_agg.pivot(index="domain", columns="date", values="appearances").fillna(0)

        if not df_sov.empty:
            most_recent_date = df_sov.columns[-1]
            df_sov = df_sov.sort_values(by=most_recent_date, ascending=False)
        
        historical_data_cache[cache_key] = (df_sov, df_metrics, df_appearances)
        return df_sov, df_metrics, df_appearances
    except Exception as e:
        logger.error(f"Error in get_total_historical_data for country '{country}': {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def compute_and_store_total_data():
    try:
        worksheet = worksheet_cache["share_of_voice"]
        rate_limit()
        data = worksheet.get_all_records()
        if not data:
            logger.warning("No data in 'share_of_voice' to compute totals")
            return
        
        df = pd.DataFrame(data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
        today = datetime.date.today().isoformat()
        df = df[(df["date"] == today) & (~df["campaign_name"].str.startswith("Total - "))]
        if df.empty:
            logger.warning(f"No non-Total data to aggregate for {today}")
            return
        
        grouped = df.groupby("country")
        for country, country_df in grouped:
            domain_sov = defaultdict(float)
            domain_appearances = defaultdict(int)
            domain_v_rank = defaultdict(list)
            domain_h_rank = defaultdict(list)
            domain_single_link = defaultdict(int)

            for _, row in country_df.iterrows():
                domain = row["domain"]
                sov = row["sov"] if pd.notna(row["sov"]) else 0
                domain_appearances[domain] += int(row["appearances"])
                v_rank = row["avg_v_rank"] if pd.notna(row["avg_v_rank"]) else 0
                h_rank = row["avg_h_rank"] if pd.notna(row["avg_h_rank"]) else 0
                domain_v_rank[domain].append(float(v_rank))
                domain_h_rank[domain].append(float(h_rank))
                domain_single_link[domain] += int(row["single_link"])
                domain_sov[domain] += float(sov)

            total_avg_v_rank = {domain: round(sum(ranks) / len(ranks), 2) for domain, ranks in domain_v_rank.items() if ranks}
            total_avg_h_rank = {domain: round(sum(ranks) / len(ranks), 2) for domain, ranks in domain_h_rank.items() if ranks}
            total_sov = sum(domain_sov.values())
            if total_sov > 0:
                domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}

            total_campaign_name = f"Total - {country}"
            save_to_db(domain_sov, domain_appearances, total_avg_v_rank, total_avg_h_rank, domain_single_link, total_campaign_name, country)
            logger.info(f"Total data computed and stored for {total_campaign_name} on {today} based on {len(country_df['campaign_name'].unique())} campaign(s)")
    except Exception as e:
        logger.error(f"Error in compute_and_store_total_data: {e}")

def create_or_update_campaign(campaign_name, job_titles, locations, country="US"):
    if not campaign_name or not job_titles or not locations or not country:
        logger.error("Missing campaign name, job titles, locations, or country")
        return False
    
    try:
        worksheet = worksheet_cache["campaigns"]
        data = worksheet.get_all_records()
        existing = next((row for row in data if row["campaign_name"] == campaign_name), None)
        
        row_data = {
            "campaign_name": campaign_name,
            "job_titles": json.dumps(job_titles),
            "locations": json.dumps(locations),
            "country": country,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        if existing:
            row_index = data.index(existing) + 2
            rate_limit()
            worksheet.update(f"A{row_index}:E{row_index}", [list(row_data.values())])
            logger.info(f"Updated campaign '{campaign_name}'")
        else:
            rate_limit()
            worksheet.append_row(list(row_data.values()))
            logger.info(f"Created campaign '{campaign_name}'")
        return True
    except Exception as e:
        logger.error(f"Error in create_or_update_campaign '{campaign_name}': {e}")
        return False

def delete_campaign(campaign_name):
    try:
        sov_worksheet = worksheet_cache["share_of_voice"]
        campaigns_worksheet = worksheet_cache["campaigns"]
        
        rate_limit()
        sov_data = sov_worksheet.get_all_records()
        sov_rows_to_keep = [row for row in sov_data if row["campaign_name"] != campaign_name]
        if len(sov_rows_to_keep) < len(sov_data):
            rate_limit()
            sov_worksheet.clear()
            sov_worksheet.append_row(["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name", "country", "single_link"])
            if sov_rows_to_keep:
                rate_limit()
                sov_worksheet.append_rows([list(row.values()) for row in sov_rows_to_keep])
            logger.info(f"Deleted records for campaign '{campaign_name}' from share_of_voice")

        rate_limit()
        campaigns_data = campaigns_worksheet.get_all_records()
        campaign_row = next((row for row in campaigns_data if row["campaign_name"] == campaign_name), None)
        if campaign_row:
            row_index = campaigns_data.index(campaign_row) + 2
            rate_limit()
            campaigns_worksheet.delete_rows(row_index)
            logger.info(f"Campaign '{campaign_name}' deleted from campaigns")
    except Exception as e:
        logger.error(f"Error deleting campaign '{campaign_name}': {e}")

def bulk_create_campaigns(df):
    try:
        grouped = df.groupby("Campaign")
        for campaign_name, group in grouped:
            job_titles = group["Keyword"].tolist()
            locations = group["Location"].fillna("").tolist()
            countries = group["Country"].fillna("US").tolist()
            country = countries[0]
            if len(job_titles) != len(locations):
                logger.error(f"Mismatch in job titles and locations for campaign '{campaign_name}'")
                continue
            if create_or_update_campaign(campaign_name, job_titles, locations, country):
                logger.info(f"Bulk created/updated campaign '{campaign_name}' with {len(job_titles)} keywords in country '{country}'")
        return True
    except Exception as e:
        logger.error(f"Error in bulk_create_campaigns: {e}")
        return False

def check_data_stored(campaign_name):
    try:
        worksheet = worksheet_cache["share_of_voice"]
        rate_limit()
        data = worksheet.get_all_records()
        today = datetime.date.today().isoformat()
        campaign_data = [row for row in data if row["campaign_name"] == campaign_name and row["date"] == today]
        if campaign_data:
            logger.info(f"✅ Check passed: Found {len(campaign_data)} rows for campaign '{campaign_name}' on {today}")
        else:
            logger.warning(f"⚠️ Check failed: No data found for campaign '{campaign_name}' on {today}")
    except Exception as e:
        logger.error(f"Error checking data for '{campaign_name}': {e}")

# Authentication function
def check_authentication():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated

def login():
    st.sidebar.header("Admin Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login", key="login_button"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.sidebar.success("Logged in successfully!")
            st.rerun()  # Refresh the app to reflect the authenticated state
        else:
            st.sidebar.error("Invalid username or password")

def logout():
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.authenticated = False
        st.rerun()

def main():
    initialize_sheets()
    st.title("Google for Jobs Visibility Tracker")

    if "refresh_campaigns" not in st.session_state:
        st.session_state.refresh_campaigns = False

    # Show login form if not authenticated
    if not check_authentication():
        login()
        st.sidebar.info("Please log in to access data fetching features.")
    else:
        logout()

    page = st.sidebar.selectbox(
        "Navigate",
        ["Visibility Tracker", "Campaign Management"],
        key="navigate_selectbox"
    )

    if page == "Visibility Tracker":
        try:
            st.image("logo.png", width=200)
        except Exception:
            st.warning("Logo not found, skipping display.")
        
        st.sidebar.header("Date Range Selector")
        today = datetime.date.today()
        default_start_date = today - datetime.timedelta(days=30)
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start_date,
            key="start_date_input"
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=today,
            key="end_date_input"
        )

        if start_date > end_date:
            st.sidebar.error("End date must be after start date!")
            return

        campaigns_data = worksheet_cache["campaigns"].get_all_records()
        campaign_names = [row["campaign_name"] for row in campaigns_data]
        countries = sorted(list(set(row.get("country", "US") for row in campaigns_data)))
        total_options = [f"Total - {country}" for country in countries]
        campaign_name_options = total_options + campaign_names
        selected_campaign_name = st.sidebar.selectbox(
            "Select Campaign",
            campaign_name_options,
            index=0,
            key="campaign_selectbox"
        )

        # Show fetch buttons only if authenticated
        if check_authentication():
            st.sidebar.header("Fetch Campaigns by Country")
            selected_country = st.sidebar.selectbox(
                "Select Country to Fetch All Campaigns",
                countries,
                key="country_fetch_selectbox"
            )
            if st.sidebar.button(f"Fetch All Campaigns for {selected_country}", key="fetch_all_country_button"):
                with st.spinner(f"Fetching data for all campaigns in {selected_country}..."):
                    success, processed_campaigns = fetch_campaigns_by_country(selected_country)
                if success:
                    st.success(f"Successfully fetched data for {len(processed_campaigns)} campaigns in {selected_country}: {', '.join(processed_campaigns)}")
                else:
                    st.error(f"Failed to fetch data for some or all campaigns in {selected_country}. Check logs for details.")

            if st.button("Fetch & Store Data", key="fetch_store_button"):
                if selected_campaign_name.startswith("Total - "):
                    compute_and_store_total_data()
                    st.success("Total data across all campaigns stored successfully!")
                else:
                    sov_data, appearances, avg_v_rank, avg_h_rank, single_link, country = compute_sov(selected_campaign_name)
                    save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, single_link, selected_campaign_name, country)
                    compute_and_store_total_data()
                    st.success(f"Data stored successfully for campaign '{selected_campaign_name}' and Totals updated!")
                historical_data_cache.clear()

        if selected_campaign_name.startswith("Total - "):
            country = selected_campaign_name.replace("Total - ", "")
            df_sov, df_metrics, df_appearances = get_total_historical_data(start_date, end_date, country)
        else:
            df_sov, df_metrics, df_appearances = get_historical_data(start_date, end_date, selected_campaign_name)

        if not df_sov.empty:
            top_domains = df_sov.iloc[:15]

            st.write("### Appearances Over Time")
            top_domains_appearances = df_appearances.loc[top_domains.index]
            fig2 = go.Figure()
            for domain in top_domains_appearances.index:
                fig2.add_trace(go.Scatter(x=top_domains_appearances.columns, y=top_domains_appearances.loc[domain], mode="markers+lines", name=domain))
            fig2.update_layout(title=f"Domain Appearances Over Time for {selected_campaign_name}", xaxis_title="Date", yaxis_title="Number of Appearances")
            st.plotly_chart(fig2)

            st.write("### Additional Metrics Over Time")
            if not df_metrics.empty:
                last_date = df_metrics.columns.get_level_values(0)[-1]
                df_metrics_sorted = df_metrics.sort_values(by=(last_date, "appearances"), ascending=False)
                st.dataframe(df_metrics_sorted.style.format("{:.2f}"))
            else:
                st.write("No additional metrics data available.")

            st.write("### Visibility Over Time")
            fig1 = go.Figure()
            for domain in top_domains.index:
                fig1.add_trace(go.Scatter(x=top_domains.columns, y=top_domains.loc[domain], mode="markers+lines", name=domain))
            fig1.update_layout(title=f"Domains Visibility Over Time for {selected_campaign_name}", xaxis_title="Date", yaxis_title="Share of Voice (%)")
            st.plotly_chart(fig1)
            st.write("#### Table of Visibility Score Data")
            st.dataframe(df_sov.style.format("{:.2f}"))
        else:
            st.write(f"No historical data available for the selected date range and {selected_campaign_name}")

    elif page == "Campaign Management":
        st.header("Campaign Management")
        
        st.subheader("Create a New Campaign")
        campaign_name = st.text_input(
            "Campaign Name (unique identifier)",
            key="campaign_name_input"
        )
        job_titles = st.text_area(
            "Job Titles (one per line)",
            height=100,
            key="job_titles_input"
        )
        locations = st.text_area(
            "Locations (one per line, matching job titles)",
            height=100,
            key="locations_input"
        )
        country = st.text_input(
            "Country",
            value="US",
            key="country_input"
        )

        if st.button("Create/Update Campaign", key="create_update_button"):
            job_titles_list = [title.strip() for title in job_titles.split('\n') if title.strip()]
            locations_list = [loc.strip() for loc in locations.split('\n') if loc.strip()]
            if len(job_titles_list) != len(locations_list):
                st.error("⚠️ The number of job titles must match the number of locations!")
            elif create_or_update_campaign(campaign_name, job_titles_list, locations_list, country):
                st.success(f"Campaign '{campaign_name}' created/updated successfully!")
                st.session_state.refresh_campaigns = True

        st.subheader("Bulk Upload Campaigns")
        uploaded_file = st.file_uploader(
            "Upload CSV (Campaign,Keyword,Location,Country)",
            type="csv",
            key="csv_uploader"
        )
        if uploaded_file and st.button("Create/Update from CSV", key="bulk_upload_button"):
            df = pd.read_csv(uploaded_file)
            if {"Campaign", "Keyword", "Location", "Country"}.issubset(df.columns):
                if bulk_create_campaigns(df):
                    st.success("All campaigns from CSV created/updated successfully!")
                    st.session_state.refresh_campaigns = True
                else:
                    st.error("Failed to process some campaigns. Check logs for details.")
            else:
                st.error("CSV must contain 'Campaign', 'Keyword', 'Location', and 'Country' columns!")

        st.subheader("Delete a Campaign")
        worksheet = worksheet_cache["campaigns"]
        campaign_names = [row["campaign_name"] for row in worksheet.get_all_records()]
        if campaign_names:
            selected_campaign_name = st.selectbox(
                "Choose a campaign to delete",
                [""] + campaign_names,
                key="delete_campaign_selectbox"
            )
            if selected_campaign_name and st.button(f"Delete {selected_campaign_name}", key=f"delete_button_{selected_campaign_name}"):
                delete_campaign(selected_campaign_name)
                st.session_state.refresh_campaigns = True
        else:
            st.write("No campaigns available to delete.")

        if st.session_state.refresh_campaigns:
            client = get_sheets_client()
            worksheet_cache["campaigns"] = get_worksheet(client, "campaigns")
            st.session_state.refresh_campaigns = False

        st.subheader("Existing Campaigns")
        campaigns = worksheet_cache["campaigns"].get_all_records()
        if campaigns:
            for campaign in campaigns:
                st.write(f"- **Campaign Name:** {campaign['campaign_name']}, **Country:** {campaign.get('country', 'US')}, **Created At:** {campaign['created_at']}")
        else:
            st.write("No campaigns created yet.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "github":
            campaign_name = sys.argv[2] if len(sys.argv) > 2 else "default"
            logger.info(f"Running automated fetch & store process for campaign: {campaign_name}")
            initialize_sheets()
            sov_data, appearances, avg_v_rank, avg_h_rank, single_link, country = compute_sov(campaign_name)
            save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, single_link, campaign_name, country)
            check_data_stored(campaign_name)
            compute_and_store_total_data()
            logger.info("Data processing completed for GitHub run")
        elif sys.argv[1] == "all_country" and len(sys.argv) > 2:
            country = sys.argv[2]
            logger.info(f"Running automated fetch & store process for all campaigns in country: {country}")
            initialize_sheets()
            success, processed_campaigns = fetch_campaigns_by_country(country)
            if success:
                logger.info(f"Successfully processed {len(processed_campaigns)} campaigns in {country}: {processed_campaigns}")
            else:
                logger.error(f"Failed to process some or all campaigns in {country}. Check logs for details.")
    else:
        main()