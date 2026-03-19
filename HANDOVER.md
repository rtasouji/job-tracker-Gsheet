# Project Handover: Google for Jobs Visibility Tracker

## 1. Overview
This tool is a simple web dashboard made with Python and **Streamlit**. It helps us see which company websites are showing up the most in Google Jobs search results (we measure this as "Share of Voice"). 

Instead of a complicated database, it just saves everything into a **Google Sheet**. Behind the scenes, it uses a service called **SerpAPI** to run the actual Google searches and collect all the job data.

### What it's built with:
*   **Language:** Python 3.9+
*   **Web Framework:** Streamlit
*   **Database:** Google Sheets (using `gspread`)
*   **Data Scraper:** SerpAPI
*   **Data crunching:** Pandas
*   **Charts & Visualisation:** Plotly

---

## 2. Project Context & Future Plans
I built this tool quite a while ago, before modern AI coding assistants were around. Because of that, and because we needed a free option at the time, I used Google Sheets to store all the data. 

**Action Required:** Now that the team has access to powerful AI coding tools, it's a great time to give this project a proper update. I highly recommend the next person uses AI to help prioritise these upgrades:
1. **Move to a real database:** It is time to swap out Google Sheets for a proper database like SQLite or a free cloud database (like Supabase or Neon). 
2. **Tidy up the code:** The main `app.py` file is quite long and does everything at once. It should be split into smaller, neater files.
3. **Make it look better:** Give the dashboard a bit of a facelift and improve the data visualisation.

---

## 3. Handover Checklist
Before you can comfortably run or edit this tool, make sure you have access to a few key things:

1.  **GitHub:** Make sure you can pull and push code to this repository.
2.  **Google Sheet:** You'll need 'Editor' rights to the main Google Sheet where our data lives.
3.  **Google Cloud Account:** You need access to the Google Cloud project that connects our app to the Google Sheet (this is where you get the 'Service Account' key from).
4.  **SerpAPI Account:** Simon has been given access to the SerpAPI dashboard where our API key lives. *Note: We are currently paying $75 per month for this service.*

---

## 4. Secret Keys & Passwords
This app needs a few secret keys and passwords to work. **Never type these directly into the code!**

Because this tool gives us a web dashboard *and* runs background tasks, you'll need to save these secrets in **two different places**:
1. **Streamlit Secrets** (or your server settings): This lets the web dashboard securely load the data for people viewing the site.
2. **GitHub Repository Secrets**: This lets the automated daily scraping jobs run successfully in the background on GitHub.

Here are the variables you need to set up (also put these in a `.env` file if you are working on your own computer):

| Name | What it is |
| :--- | :--- |
| `GOOGLE_SHEETS_CREDS` | The long JSON string from our Google Service Account. Make sure it stays formatted as one single line if you copy it into a server. |
| `SPREADSHEET_ID` | The messy string of letters in the middle of our Google Sheets URL. |
| `SERP_API_KEY` | Our secret API key from SerpAPI.com. |
| `ADMIN_USERNAME` | The username you'll use to log into the "Campaign Management" panel on the dashboard. |
| `ADMIN_PASSWORD` | The password you'll use for that same admin panel. |

---

## 5. Local Setup
Want to run and test the code on your own computer? Just open your terminal and follow these steps:

### Step 1: Download the code
```bash
git clone https://github.com/rtasouji/job-tracker-Gsheet.git
cd job-tracker-Gsheet
```

### Step 2: Set up Python
```bash
python -m venv venv

# If you use Windows, run this:
venv\Scripts\activate

# If you use macOS or Linux, run this:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Add your secrets
Set up your `.env` file with the secret keys from Section 4.

### Step 4: Run the app!
```bash
streamlit run app.py
```
A browser window should pop up automatically at `http://localhost:8501`.

---

## 6. Making It Live (Deployment)
The absolute easiest and completely free way to put this app on the internet is by using **Streamlit Community Cloud**.

### How to get it online:
1.  Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
2.  Click **New App**.
3.  Choose this repository (`job-tracker-Gsheet`) and make sure the main file path says `app.py`.
4.  Before you click Deploy, click on **Advanced Settings** and paste all of your Secret Keys (from Section 4) into the text box using the TOML format. It should look like this:
    ```toml
    GOOGLE_SHEETS_CREDS = '{"type": "service_account", ...}'
    SPREADSHEET_ID = "your_sheet_id_here"
    SERP_API_KEY = "your_serp_api_key"
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "secure_password"
    ```
5.  Click **Deploy**. Streamlit will do the rest and give you a public URL (like `https://job-tracker-gsheet.streamlit.app`) that you can share with the team!

---

## 7. Our Google Sheet Layout
**Live Database Link:** [GfJ Tracker Database](https://docs.google.com/spreadsheets/d/1VplsnX77pdvPffT3CgtjHjwky_cC3w-8yZeaUt2-UC0/edit?gid=0#gid=0)

The code actually manages the Google Sheet completely by itself. It looks for two specific tabs, and if they aren't there, it will try to create them:

1.  **`campaigns` tab:** This is where it remembers what jobs we want to search for.
2.  **`share_of_voice` tab:** This is the massive history book where it logs the daily visibility scores for every website it finds.

> **⚠️ A quick warning:** Please don't manually delete rows, sort the columns, or leave blank spaces anywhere in the data while the script is running, as it might confuse the code and break the layout!

---

## 8. Automated Daily Tracking
This tool is clever enough to fetch fresh job data in the background every day without anyone opening the app. You can trigger it from the command line, which makes it perfect for **GitHub Actions** or scheduled tasks.

*   **To get data for one specific campaign:**
    ```bash
    python app.py github <campaign_name>
    ```
*   **To get data for a whole country at once:**
    ```bash
    python app.py all_country <country_code>
    ```

**Checking it worked:**
There is a tiny script called `check_db.py`. If you run it after a daily scrape, it simply logs into Google Sheets to double-check that today's data was actually saved properly.

---

## 9. Good Things to Know (Technical Debt)
As you take over the project, there are two main technical quirks you should keep in mind:

1. **Google Sheets gets slow:** Saving thousands of rows into a Google Sheet every single day will eventually slow down the dashboard and hit Google's connection limits. As we mentioned in the Future Plans section, moving to a real database is the best way to fix this permanently.
2. **SerpAPI usage costs:** To make scraping faster, the app now checks up to 10 jobs at the exact same time. However, it doesn't remember old results. This means if you run the script twice in one day, it will essentially double our SerpAPI bill for that day, so please be careful with manual runs!
