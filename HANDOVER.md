# Project Handover: Google for Jobs Visibility Tracker

## 1. Overview
This project is a Python-based web application built with **Streamlit** that tracks the "Share of Voice" (SoV) and visibility of various domains in Google Jobs search results. 

Instead of a traditional database, the application utilises **Google Sheets** to store historical data and campaign configurations. It uses **SerpAPI** to run Google Jobs searches and aggregate the results.

### Tech Stack
*   **Language:** Python 3.9+
*   **Frontend/App Framework:** Streamlit
*   **Database:** Google Sheets (via `gspread` and `oauth2client`)
*   **Data Scraper:** SerpAPI (`requests`, `tenacity`)
*   **Data Processing:** Pandas
*   **Data Visualisation:** Plotly

---

## 2. Project Context & Future Roadmap
This tool was built prior to the widespread availability of modern AI coding assistants (like Claude Code, Antigravity, etc.). As a result, the architecture relies on older paradigms. Specifically, **Google Sheets was chosen as the database simply because it was the only fully free and easy option available at the time**, not because it is the optimal technical choice for scale.

**Action Required:** Now that powerful AI coding tools are available to the team, this project is uniquely positioned for a rapid, AI-assisted overhaul. It is highly recommended that the new owner uses these tools to prioritise the following:
1. **Database Migration:** Move away from Google Sheets to a robust database (e.g., SQLite, or a free cloud PostgreSQL like Supabase/Neon).
2. **Codebase Refactor:** Split the monolithic `app.py` into modular components.
3. **UI/UX Enhancements:** Improve the data visualisation and dashboard design.

---

## 3. Handover Checklist
To successfully take over this project, you must ensure you have access to the following accounts and credentials:

1.  **GitHub Repository:** Ensure you have pull/push access to this repository.
2.  **Google Sheet (Database):** You need `Editor` or `Owner` permissions on the master Google Sheet used by the application.
3.  **Google Cloud Platform (GCP) Service Account:** You need access to the GCP project where the Google Sheets API is enabled. You will need to generate or access the JSON key for the Service Account that the app uses.
4.  **SerpAPI Account:** You need the login details for the SerpAPI account to manage billing and access the API key used for scraping jobs.

---

## 4. Environment Variables & Secrets
The application relies heavily on environment variables to function securely. **Do not hardcode these in the source code.**

Because this application acts as both a web dashboard and a background job runner, you will need to configure these secrets in **two different places**:
1. **Streamlit Secrets** (or your hosting provider's environment variables): This is required for the web dashboard to function so users can view the data.
2. **GitHub Repository Secrets**: This is required for the automated daily scraping jobs (GitHub Actions) to run successfully in the background.

You need to configure the following variables in both locations (and your local `.env` file if developing locally):

| Variable Name           | Description / Source                                                                                                                                           |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GOOGLE_SHEETS_CREDS`   | The raw JSON string of the Google Service Account credentials. *Note: Ensure this is formatted correctly as a single string if deployed.*                      |
| `SPREADSHEET_ID`        | The ID of the Google Sheet (found in the URL between `/d/` and `/edit`).                                                                                       |
| `SERP_API_KEY`          | Your secret API key from SerpAPI.com.                                                                                                                          |
| `ADMIN_USERNAME`        | The username required to log into the "Campaign Management" panel in the UI. (Default in code if missing is "admin", but you should override this).            |
| `ADMIN_PASSWORD`        | The password required to log into the "Campaign Management" panel.                                                                                             |

---

## 5. Local Development Setup
Follow these steps to run the application on your local machine.

### Step 1: Clone the repository
```bash
git clone https://github.com/rtasouji/job-tracker-Gsheet.git
cd job-tracker-Gsheet
```

### Step 2: Create a virtual environment and install dependencies
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Set up environment variables
Depending on your OS, export the variables listed in Section 4. Alternatively, you can run the app with variables inline (Linux/macOS) or use a `.env` file if you install `python-dotenv` (requires modifying `app.py` slightly).

### Step 4: Run the Streamlit application
```bash
streamlit run app.py
```
The application should now be accessible at `http://localhost:8501`.

---

## 6. Google Sheets Database Structure
The application automatically manages the Google Sheet, but it expects two specific worksheets (tabs) to exist. If they do not exist, the app will attempt to create them.

1.  **`campaigns` tab:** Stores the configurations for what jobs to search. Columns include `campaign_name`, `job_titles` (JSON array), `locations` (JSON array), `country`, and `created_at`.
2.  **`share_of_voice` tab:** The historical database where the script writes daily metrics. Columns include `domain`, `date`, `sov`, `appearances`, `avg_v_rank`, `avg_h_rank`, `campaign_name`, `country`, and `single_link`.

> **⚠️ Important Warning regarding manual edits:** The app reads the whole sheet and appends rows based on index calculations. Manually deleting rows, sorting columns, or leaving blank spaces in the Google Sheet while the script is running might result in data corruption. 

---

## 7. Automation & Background Jobs
The system is designed to run automated daily fetches without loading the web UI. This is triggered via command-line arguments, which is ideal for **GitHub Actions** or **Cron Jobs**.

*   **Fetch a specific campaign:** 
    ```bash
    python app.py github <campaign_name>
    ```
*   **Fetch all campaigns for a specific country:**
    ```bash
    python app.py all_country <country_code>
    ```

### Validation Script
There is a utility script included called `check_db.py`. This script is meant to be run immediately after the automated data fetching job. It checks the Google Sheet to ensure that rows for *today's date* were successfully written. 
```bash
python check_db.py <campaign_name>
```
If data is missing, it logs a warning/error.

---

## 8. Known Architectural Limitations
As the new owner, please be aware of the following technical debt:
1. **Google Sheets Limits:** Fetching and writing thousands of rows daily to Google Sheets will eventually hit API payload limits and slow down the dashboard significantly. As mentioned in the Roadmap, migration to a real database (SQLite/PostgreSQL) is highly recommended.
2. **SerpAPI Costs:** To speed up scraping, job queries are now processed concurrently using a `ThreadPoolExecutor` (max 10 workers). However, the app still does not cache raw SerpAPI responses locally. Running the same fetch script twice in one day will double your SerpAPI usage, so caution is advised when doing manual runs.
