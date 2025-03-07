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

# Parse CREDS_JSON
if CREDS_JSON_RAW:
    try:
        CREDS_JSON = json.loads(CREDS_JSON_RAW)
        logger.info(f"Successfully parsed GOOGLE_SHEETS_CREDS into dict with keys: {list(CREDS_JSON.keys())}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GOOGLE_SHEETS_CREDS: {e}")
        CREDS_JSON = None
else:
    CREDS_JSON = None

# Validate environment variables
if not CREDS_JSON or not SPREADSHEET_ID or not SERP_API_KEY:
    missing_vars = [var for var, val in [("GOOGLE_SHEETS_CREDS", CREDS_JSON), ("SPREADSHEET_ID", SPREADSHEET_ID), ("SERP_API_KEY", SERP_API_KEY)] if not val]
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

def get_sheets_client():
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(CREDS_JSON, SCOPE)
        logger.info("Sheets client created successfully")
        return gspread.authorize(creds)
    except Exception as e:
        logger.error(f"Error creating Sheets client: {e}")
        raise

def get_worksheet(client, sheet_name):
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        logger.info(f"Accessed existing worksheet: {sheet_name}")
        return worksheet
    except gspread.WorksheetNotFound:
        logger.info(f"Worksheet {sheet_name} not found, creating it")
        return spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=10)

def initialize_sheets():
    try:
        client = get_sheets_client()
        get_worksheet(client, "share_of_voice")
        get_worksheet(client, "campaigns")
        logger.info("Google Sheets initialized")
    except Exception as e:
        logger.error(f"Error initializing Sheets: {e}")
        raise

# Load jobs from Google Sheets
def load_jobs(campaign_name):
    try:
        client = get_sheets_client()
        worksheet = get_worksheet(client, "campaigns")
        data = worksheet.get_all_records()
        campaign_data = next((row for row in data if row["campaign_name"] == campaign_name), None)
        
        if not campaign_data:
            logger.warning(f"Campaign '{campaign_name}' not found")
            return []
        
        job_titles = json.loads(campaign_data["job_titles"]) if campaign_data["job_titles"] else []
        locations = json.loads(campaign_data["locations"]) if campaign_data["locations"] else []
        
        if len(job_titles) != len(locations):
            logger.error(f"Mismatch between job titles ({len(job_titles)}) and locations ({len(locations)}) for campaign '{campaign_name}'")
            return []
        
        logger.info(f"Loaded {len(job_titles)} job queries for campaign '{campaign_name}'")
        return [{"job_title": title, "location": loc} for title, loc in zip(job_titles, locations)]
    except Exception as e:
        logger.error(f"Error in load_jobs for campaign '{campaign_name}': {e}")
        return []

# Fetch Google Jobs Results from SerpAPI
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
        
        response = requests.get(url, params=params)
        logger.info(f"SerpAPI response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs_results", [])
        logger.info(f"Fetched {len(jobs)} jobs for query: '{query}'")
        return jobs
    except requests.RequestException as e:
        logger.error(f"SerpAPI request failed: {e}")
        return []

# Compute Share of Voice
def compute_sov(campaign_name):
    logger.info(f"Starting compute_sov for campaign '{campaign_name}'")
    domain_sov = defaultdict(float)
    domain_appearances = defaultdict(int)
    domain_v_rank = defaultdict(list)
    domain_h_rank = defaultdict(list)
    total_sov = 0

    jobs_data = load_jobs(campaign_name)
    if not jobs_data:
        logger.warning(f"No job data for campaign '{campaign_name}'")
        return {}, {}, {}, {}

    for job_query in jobs_data:
        job_title = job_query["job_title"]
        location = job_query["location"]
        jobs = get_google_jobs_results(job_title, location)
        for job_rank, job in enumerate(jobs, start=1):
            apply_options = job.get("apply_options", [])
            V = 1 / job_rank
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
        time.sleep(1)  # Throttle SerpAPI requests

    if total_sov > 0:
        domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}
    
    domain_avg_v_rank = {domain: round(sum(vr) / len(vr), 2) for domain, vr in domain_v_rank.items() if vr}
    domain_avg_h_rank = {domain: round(sum(hr) / len(hr), 2) for domain, hr in domain_h_rank.items() if hr}
    
    logger.info(f"Computed SoV for '{campaign_name}': {len(domain_sov)} domains")
    return domain_sov, domain_appearances, domain_avg_v_rank, domain_avg_h_rank

# Extract Domain from URL
def extract_domain(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    return domain.lower().replace("www.", "")

# Save data to Google Sheets
def save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, campaign_name):
    if not sov_data:
        logger.warning(f"No SoV data to save for campaign '{campaign_name}'")
        return
    
    try:
        client = get_sheets_client()
        worksheet = get_worksheet(client, "share_of_voice")
        today = datetime.date.today().isoformat()
        rows = [[domain, today, round(sov_data[domain], 2), appearances[domain], 
                 avg_v_rank.get(domain, 0), avg_h_rank.get(domain, 0), campaign_name]
                for domain in sov_data]
        worksheet.append_rows(rows)
        logger.info(f"Saved {len(rows)} rows to 'share_of_voice' for campaign '{campaign_name}'")
    except Exception as e:
        logger.error(f"Error saving to Google Sheets for campaign '{campaign_name}': {e}")

# Retrieve historical data
def get_historical_data(start_date, end_date, campaign_name):
    try:
        client = get_sheets_client()
        worksheet = get_worksheet(client, "share_of_voice")
        data = worksheet.get_all_records()
        df = pd.DataFrame(data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name"])
        
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
            "avg_h_rank": "mean"
        })
        
        df_sov = df_agg.pivot(index="domain", columns="date", values="sov").fillna(0)
        df_metrics = df_agg.pivot(index="domain", columns="date", values=["appearances", "avg_v_rank", "avg_h_rank"]).fillna(0)
        df_metrics = df_metrics.swaplevel(axis=1).sort_index(axis=1)
        df_appearances = df_agg.pivot(index="domain", columns="date", values="appearances").fillna(0)

        if not df_sov.empty:
            most_recent_date = df_sov.columns[-1]
            df_sov = df_sov.sort_values(by=most_recent_date, ascending=False)
        
        return df_sov, df_metrics, df_appearances
    except Exception as e:
        logger.error(f"Error in get_historical_data for '{campaign_name}': {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Total historicalGather historical data for all campaigns data
def get_total_historical_data(start_date, end_date):
    try:
        client = get_sheets_client()
        worksheet = get_worksheet(client, "share_of_voice")
        data = worksheet.get_all_records()
        df = pd.DataFrame(data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name"])
        
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        
        if df.empty:
            logger.info(f"No total historical data in range {start_date} to {end_date}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_agg = df.groupby(["domain", "date"], as_index=False).agg({
            "sov": "sum",
            "appearances": "sum",
            "avg_v_rank": "mean",
            "avg_h_rank": "mean"
        })
        
        df_sov = df_agg.pivot(index="domain", columns="date", values="sov").fillna(0)
        df_metrics = df_agg.pivot(index="domain", columns="date", values=["appearances", "avg_v_rank", "avg_h_rank"]).fillna(0)
        df_metrics = df_metrics.swaplevel(axis=1).sort_index(axis=1)
        df_appearances = df_agg.pivot(index="domain", columns="date", values="appearances").fillna(0)

        if not df_sov.empty:
            most_recent_date = df_sov.columns[-1]
            df_sov = df_sov.sort_values(by=most_recent_date, ascending=False)
        
        return df_sov, df_metrics, df_appearances
    except Exception as e:
        logger.error(f"Error in get_total_historical_data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Compute and store total data across all campaigns
def compute_and_store_total_data():
    try:
        client = get_sheets_client()
        worksheet = get_worksheet(client, "share_of_voice")
        data = worksheet.get_all_records()
        if not data:
            logger.warning("No data in 'share_of_voice' to compute totals")
            return
        
        df = pd.DataFrame(data, columns=["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name"])
        
        domain_sov = defaultdict(float)
        domain_appearances = defaultdict(int)
        domain_v_rank = defaultdict(list)
        domain_h_rank = defaultdict(list)

        for _, row in df.iterrows():
            domain = row["domain"]
            domain_sov[domain] += float(row["sov"])
            domain_appearances[domain] += int(row["appearances"])
            domain_v_rank[domain].append(float(row["avg_v_rank"]))
            domain_h_rank[domain].append(float(row["avg_h_rank"]))

        total_avg_v_rank = {domain: round(sum(ranks) / len(ranks), 2) for domain, ranks in domain_v_rank.items() if ranks}
        total_avg_h_rank = {domain: round(sum(ranks) / len(ranks), 2) for domain, ranks in domain_h_rank.items() if ranks}
        total_sov = sum(domain_sov.values())
        if total_sov > 0:
            domain_sov = {domain: round((sov / total_sov) * 100, 4) for domain, sov in domain_sov.items()}

        save_to_db(domain_sov, domain_appearances, total_avg_v_rank, total_avg_h_rank, "Total")
        logger.info("Total data computed and stored successfully")
    except Exception as e:
        logger.error(f"Error in compute_and_store_total_data: {e}")

# Create or update campaign
def create_or_update_campaign(campaign_name, job_titles, locations):
    if not campaign_name or not job_titles or not locations:
        logger.error("Missing campaign name, job titles, or locations")
        return False
    
    try:
        client = get_sheets_client()
        worksheet = get_worksheet(client, "campaigns")
        data = worksheet.get_all_records()
        existing = next((row for row in data if row["campaign_name"] == campaign_name), None)
        
        row_data = {
            "campaign_name": campaign_name,
            "job_titles": json.dumps(job_titles),
            "locations": json.dumps(locations),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        if existing:
            row_index = data.index(existing) + 2  # +2 for header
            worksheet.update(f"A{row_index}:D{row_index}", [list(row_data.values())])
            logger.info(f"Updated campaign '{campaign_name}'")
        else:
            worksheet.append_row(list(row_data.values()))
            logger.info(f"Created campaign '{campaign_name}'")
        return True
    except Exception as e:
        logger.error(f"Error in create_or_update_campaign '{campaign_name}': {e}")
        return False

# Delete campaign
def delete_campaign(campaign_name):
    try:
        client = get_sheets_client()
        sov_worksheet = get_worksheet(client, "share_of_voice")
        campaigns_worksheet = get_worksheet(client, "campaigns")
        
        sov_data = sov_worksheet.get_all_records()
        sov_rows_to_keep = [row for row in sov_data if row["campaign_name"] != campaign_name]
        if len(sov_rows_to_keep) < len(sov_data):
            sov_worksheet.clear()
            sov_worksheet.append_row(["domain", "date", "sov", "appearances", "avg_v_rank", "avg_h_rank", "campaign_name"])
            if sov_rows_to_keep:
                sov_worksheet.append_rows([list(row.values()) for row in sov_rows_to_keep])
            logger.info(f"Deleted records for campaign '{campaign_name}' from share_of_voice")

        campaigns_data = campaigns_worksheet.get_all_records()
        campaign_row = next((row for row in campaigns_data if row["campaign_name"] == campaign_name), None)
        if campaign_row:
            row_index = campaigns_data.index(campaign_row) + 2
            campaigns_worksheet.delete_rows(row_index)
            logger.info(f"Campaign '{campaign_name}' deleted from campaigns")
    except Exception as e:
        logger.error(f"Error deleting campaign '{campaign_name}': {e}")

# Bulk create campaigns from CSV
def bulk_create_campaigns(df):
    try:
        grouped = df.groupby("Campaign")
        for campaign_name, group in grouped:
            job_titles = group["Keyword"].tolist()
            locations = group["Location"].fillna("").tolist()
            if len(job_titles) != len(locations):
                logger.error(f"Mismatch in job titles and locations for campaign '{campaign_name}'")
                continue
            if create_or_update_campaign(campaign_name, job_titles, locations):
                logger.info(f"Bulk created/updated campaign '{campaign_name}' with {len(job_titles)} keywords")
        return True
    except Exception as e:
        logger.error(f"Error in bulk_create_campaigns: {e}")
        return False

# Streamlit UI
def main():
    initialize_sheets()
    st.title("Google for Jobs Visibility Tracker")
    page = st.sidebar.selectbox("Navigate", ["Visibility Tracker", "Campaign Management"])

    if page == "Visibility Tracker":
        try:
            st.image("logo.png", width=200)
        except Exception:
            st.warning("Logo not found, skipping display.")
        
        st.sidebar.header("Date Range Selector")
        today = datetime.date.today()
        default_start_date = today - datetime.timedelta(days=30)
        start_date = st.sidebar.date_input("Start Date", value=default_start_date)
        end_date = st.sidebar.date_input("End Date", value=today)

        if start_date > end_date:
            st.sidebar.error("End date must be after start date!")
            return

        client = get_sheets_client()
        worksheet = get_worksheet(client, "campaigns")
        campaign_names = [row["campaign_name"] for row in worksheet.get_all_records()]
        campaign_name_options = ["Total"] + campaign_names
        selected_campaign_name = st.sidebar.selectbox("Select Campaign", campaign_name_options, index=0)

        if st.button("Fetch & Store Data"):
            if selected_campaign_name == "Total":
                compute_and_store_total_data()
                st.success("Total data across all campaigns stored successfully!")
            else:
                sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov(selected_campaign_name)
                save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, selected_campaign_name)
                st.success(f"Data stored successfully for campaign '{selected_campaign_name}'!")

        st.write("### Visibility Over Time")
        if selected_campaign_name == "Total":
            df_sov, df_metrics, df_appearances = get_total_historical_data(start_date, end_date)
        else:
            df_sov, df_metrics, df_appearances = get_historical_data(start_date, end_date, selected_campaign_name)

        if not df_sov.empty:
            top_domains = df_sov.iloc[:15]
            fig1 = go.Figure()
            for domain in top_domains.index:
                fig1.add_trace(go.Scatter(x=top_domains.columns, y=top_domains.loc[domain], mode="markers+lines", name=domain))
            fig1.update_layout(title=f"Domains Visibility Over Time for {selected_campaign_name}", xaxis_title="Date", yaxis_title="Share of Voice (%)")
            st.plotly_chart(fig1)
            st.write("#### Table of Visibility Score Data")
            st.dataframe(df_sov.style.format("{:.2f}"))

            st.write("### Appearances Over Time")
            top_domains_appearances = df_appearances.loc[top_domains.index]
            fig2 = go.Figure()
            for domain in top_domains_appearances.index:
                fig2.add_trace(go.Scatter(x=top_domains_appearances.columns, y=top_domains_appearances.loc[domain], mode="markers+lines", name=domain))
            fig2.update_layout(title=f"Domain Appearances Over Time for {selected_campaign_name}", xaxis_title="Date", yaxis_title="Number of Appearances")
            st.plotly_chart(fig2)
            st.write("### Additional Metrics Over Time")
            st.dataframe(df_metrics.style.format("{:.2f}"))
        else:
            st.write(f"No historical data available for the selected date range and {selected_campaign_name}")

    elif page == "Campaign Management":
        st.header("Campaign Management")
        
        # Manual Campaign Entry
        st.subheader("Create a New Campaign")
        campaign_name = st.text_input("Campaign Name (unique identifier)")
        job_titles = st.text_area("Job Titles (one per line)", height=100)
        locations = st.text_area("Locations (one per line, matching job titles)", height=100)

        if st.button("Create/Update Campaign"):
            job_titles_list = [title.strip() for title in job_titles.split('\n') if title.strip()]
            locations_list = [loc.strip() for loc in locations.split('\n') if loc.strip()]
            if len(job_titles_list) != len(locations_list):
                st.error("⚠️ The number of job titles must match the number of locations!")
            elif create_or_update_campaign(campaign_name, job_titles_list, locations_list):
                st.success(f"Campaign '{campaign_name}' created/updated successfully!")

        # Bulk Campaign Upload
        st.subheader("Bulk Upload Campaigns")
        uploaded_file = st.file_uploader("Upload CSV (Campaign,Keyword,Location)", type="csv")
        if uploaded_file and st.button("Create/Update from CSV"):
            df = pd.read_csv(uploaded_file)
            if {"Campaign", "Keyword", "Location"}.issubset(df.columns):
                if bulk_create_campaigns(df):
                    st.success("All campaigns from CSV created/updated successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to process some campaigns. Check logs for details.")
            else:
                st.error("CSV must contain 'Campaign', 'Keyword', and 'Location' columns!")

        # Delete Campaign
        st.subheader("Delete a Campaign")
        client = get_sheets_client()
        worksheet = get_worksheet(client, "campaigns")
        campaign_names = [row["campaign_name"] for row in worksheet.get_all_records()]
        if campaign_names:
            selected_campaign_name = st.selectbox("Choose a campaign to delete", [""] + campaign_names)
            if selected_campaign_name and st.button(f"Delete {selected_campaign_name}"):
                delete_campaign(selected_campaign_name)
                st.experimental_rerun()
        else:
            st.write("No campaigns available to delete.")

        # Existing Campaigns
        st.subheader("Existing Campaigns")
        campaigns = worksheet.get_all_records()
        if campaigns:
            for campaign in campaigns:
                st.write(f"- **Campaign Name:** {campaign['campaign_name']}, **Created At:** {campaign['created_at']}")
        else:
            st.write("No campaigns created yet.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "github":
        campaign_name = sys.argv[2] if len(sys.argv) > 2 else "default"
        logger.info(f"Running automated fetch & store process for campaign: {campaign_name}")
        initialize_sheets()
        sov_data, appearances, avg_v_rank, avg_h_rank = compute_sov(campaign_name)
        save_to_db(sov_data, appearances, avg_v_rank, avg_h_rank, campaign_name)
        compute_and_store_total_data()
        logger.info("Data stored successfully!")
    else:
        main()