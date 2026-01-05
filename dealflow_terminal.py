import streamlit as st
import pandas as pd
import openai
import os
import json
from datetime import datetime
from exa_py import Exa
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Allowed email domain for access (change this to your company domain)
ALLOWED_EMAIL_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "pacecapital.com")

# ============================================================================
# SUPABASE CLIENT
# ============================================================================

def get_supabase_client() -> Client:
    """Get Supabase client with credentials from secrets or environment."""
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        return None
    
    return create_client(url, key)


def is_supabase_configured() -> bool:
    """Check if Supabase is properly configured."""
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
    return bool(url and key)


# ============================================================================
# AUTHENTICATION
# ============================================================================

def check_email_domain(email: str) -> bool:
    """Check if email belongs to the allowed domain."""
    if not email:
        return False
    return email.lower().endswith(f"@{ALLOWED_EMAIL_DOMAIN.lower()}")


def show_confidential_page():
    """Display the confidential information page for unauthorized users."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100;200;300;400;500&display=swap');
        
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        .confidential-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
            text-align: center;
            font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
        }
        
        .confidential-icon {
            font-size: 64px;
            margin-bottom: 24px;
        }
        
        .confidential-title {
            font-size: 28px;
            font-weight: 400;
            color: #FF4444;
            margin-bottom: 16px;
            letter-spacing: 3px;
        }
        
        .confidential-message {
            font-size: 14px;
            color: #888888;
            max-width: 400px;
            line-height: 1.6;
            margin-bottom: 32px;
        }
        
        .confidential-domain {
            font-size: 12px;
            color: #444444;
            border: 1px solid #333333;
            padding: 8px 16px;
            margin-top: 16px;
        }
    </style>
    
    <div class="confidential-container">
        <div class="confidential-icon">🔒</div>
        <div class="confidential-title">CONFIDENTIAL</div>
        <div class="confidential-message">
            This information is proprietary and confidential.<br><br>
            Access is restricted to authorized Pace Capital team members only.
        </div>
        <div class="confidential-domain">
            Authorized domain: @""" + ALLOWED_EMAIL_DOMAIN + """
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_login_page():
    """Display the login page with Google OAuth."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100;200;300;400;500&display=swap');
        
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
            padding-top: 15vh;
            margin-bottom: 24px;
        }
        
        .login-title {
            font-size: 24px;
            font-weight: 400;
            color: #FFFFFF;
            margin-bottom: 8px;
            letter-spacing: 2px;
        }
        
        .login-subtitle {
            font-size: 12px;
            color: #666666;
            margin-bottom: 32px;
        }
        
        .login-message {
            font-size: 13px;
            color: #888888;
            max-width: 350px;
            line-height: 1.6;
            margin-bottom: 24px;
        }
        
        .domain-note {
            font-size: 11px;
            color: #444444;
            margin-top: 16px;
        }
    </style>
    
    <div class="login-container">
        <div class="login-title">M&A_TRACKER_V1.0</div>
        <div class="login-subtitle">by Pace Capital</div>
        <div class="login-message">
            Sign in with your Pace Capital Google account to access the M&A deal tracker.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Simple email-based authentication
        # In production, replace with proper Google OAuth
        email = st.text_input("Email", placeholder=f"you@{ALLOWED_EMAIL_DOMAIN}", key="login_email")
        
        if st.button("[ SIGN IN ]", key="sign_in_btn", use_container_width=True):
            if email:
                if check_email_domain(email):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error(f"Access denied. Please use a @{ALLOWED_EMAIL_DOMAIN} email.")
            else:
                st.warning("Please enter your email address.")
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 16px;">
            <span style="font-size: 11px; color: #444444; font-family: 'Geist Mono', monospace;">
                Access restricted to @{ALLOWED_EMAIL_DOMAIN}
            </span>
        </div>
        """, unsafe_allow_html=True)


def init_auth():
    """Initialize authentication state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None


def logout():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.rerun()


# ============================================================================
# PAGE CONFIG AND STYLES
# ============================================================================

st.set_page_config(
    page_title="M&A Tracker | Pace Capital",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Terminal-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100;200;300;400;500&display=swap');
    
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .main .block-container {
        background-color: #000000;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Apply font to main content but exclude dataframe internals */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp label, 
    .stApp .stMarkdown, .stApp .stText,
    .stTextInput input, .stTextArea textarea,
    .stButton button, .stSelectbox, .stNumberInput {
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace !important;
        color: #FFFFFF !important;
        font-size: 12px !important;
        font-weight: 300 !important;
    }
    
    h1 {
        color: #888888 !important;
        font-weight: 400 !important;
        letter-spacing: 2px;
        margin-bottom: 0 !important;
        font-size: 22px !important;
    }
    
    .byline {
        color: #666666;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
        font-size: 10px !important;
        font-weight: 300 !important;
        margin-top: 0;
        letter-spacing: 1px;
    }
    
    .blink {
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .stTextInput > div > div > input {
        background-color: #000000;
        border: 1px solid #444444;
        color: #FFFFFF;
        border-radius: 0px;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace !important;
        font-size: 12px !important;
        font-weight: 300 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid #00FF00;
        box-shadow: 0 0 3px #00FF00;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #000000;
        border: 1px solid #444444;
        color: #FFFFFF;
        border-radius: 0px;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace !important;
        font-size: 12px !important;
        font-weight: 300 !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 1px solid #00FF00;
        box-shadow: 0 0 3px #00FF00;
    }
    
    button {
        background-color: #000000 !important;
        border: 1px solid #444444 !important;
        color: #AAAAAA !important;
        border-radius: 0px !important;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace !important;
        font-size: 11px !important;
        font-weight: 300 !important;
        padding: 0.4rem 0.8rem !important;
    }
    
    button:hover {
        border-color: #00FF00 !important;
        color: #00FF00 !important;
        box-shadow: 0 0 3px #00FF00 !important;
    }
    
    .stDataFrame {
        background-color: #000000;
        border: 1px solid #333333;
        font-size: 11px !important;
    }
    
    .stDataFrame td, .stDataFrame th {
        font-size: 11px !important;
        font-weight: 300 !important;
        padding: 4px 8px !important;
    }
    
    .metric-container {
        border: 1px solid #333333;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: #000000;
    }
    
    .success-message {
        color: #00FF00;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
        font-size: 11px;
        font-weight: 300;
        border: 1px solid #00FF00;
        padding: 0.4rem;
        margin: 0.8rem 0;
    }
    
    .error-message {
        color: #FF0000;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
        font-size: 11px;
        font-weight: 300;
        border: 1px solid #FF0000;
        padding: 0.4rem;
        margin: 0.8rem 0;
    }
    
    .warning-message {
        color: #FFAA00;
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
        font-size: 11px;
        font-weight: 300;
        border: 1px solid #FFAA00;
        padding: 0.4rem;
        margin: 0.8rem 0;
    }
    
    /* Streamlit info/warning/success boxes */
    .stAlert {
        font-size: 11px !important;
        font-weight: 300 !important;
    }
    
    .user-badge {
        font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace;
        font-size: 10px;
        color: #00FF00;
        border: 1px solid #00FF00;
        padding: 4px 8px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FUNCTIONS (SUPABASE + CSV FALLBACK)
# ============================================================================

# CSV file path for fallback
CSV_FILE = "mergers.csv"

# Required columns
REQUIRED_COLUMNS = [
    "Company (Target)", "Acquirer", "Category", "Amount", "Revenue", "Multiple", "Rationale",
    "Company URL", "Company Description", "Last Round Raised", "Valuation of Last Round", "Total Raised",
    "Date Added"
]

# Database column mapping
DB_TO_CSV_COLUMNS = {
    "company_target": "Company (Target)",
    "acquirer": "Acquirer",
    "category": "Category",
    "amount": "Amount",
    "revenue": "Revenue",
    "multiple": "Multiple",
    "rationale": "Rationale",
    "company_url": "Company URL",
    "company_description": "Company Description",
    "last_round_raised": "Last Round Raised",
    "valuation_of_last_round": "Valuation of Last Round",
    "total_raised": "Total Raised",
    "date_added": "Date Added",
}

CSV_TO_DB_COLUMNS = {v: k for k, v in DB_TO_CSV_COLUMNS.items()}


def load_data_from_supabase() -> pd.DataFrame:
    """Load data from Supabase."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return None
        
        response = supabase.table("deals").select("*").order("date_added", desc=True).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            # Rename columns to match CSV format
            df = df.rename(columns=DB_TO_CSV_COLUMNS)
            # Select only required columns
            available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
            df = df[available_cols]
            return df
        else:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
    except Exception as e:
        st.warning(f"Supabase error: {e}. Falling back to local CSV.")
        return None


def load_data_from_csv() -> pd.DataFrame:
    """Load data from local CSV file."""
    if os.path.exists(CSV_FILE):
        try:
            if os.path.getsize(CSV_FILE) == 0:
                return pd.DataFrame(columns=REQUIRED_COLUMNS)
            
            df = pd.read_csv(CSV_FILE)
            
            if df.empty:
                return pd.DataFrame(columns=REQUIRED_COLUMNS)
            
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = "N/A"
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)


def load_data() -> pd.DataFrame:
    """Load data from Supabase, with CSV fallback."""
    if is_supabase_configured():
        df = load_data_from_supabase()
        if df is not None:
            return df
    
    return load_data_from_csv()


def save_data_to_supabase(df: pd.DataFrame) -> bool:
    """Save a single new row to Supabase (for appending)."""
    # This is handled differently - see save_deal_to_supabase
    pass


def save_deal_to_supabase(deal_data: dict) -> bool:
    """Save a single deal to Supabase."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        # Convert to database column names
        db_data = {}
        for csv_col, db_col in CSV_TO_DB_COLUMNS.items():
            if csv_col in deal_data:
                db_data[db_col] = deal_data[csv_col]
        
        supabase.table("deals").insert(db_data).execute()
        return True
    except Exception as e:
        st.error(f"Supabase insert error: {e}")
        return False


def update_deal_in_supabase(deal_id: int, deal_data: dict) -> bool:
    """Update a deal in Supabase."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        # Convert to database column names
        db_data = {}
        for csv_col, db_col in CSV_TO_DB_COLUMNS.items():
            if csv_col in deal_data:
                db_data[db_col] = deal_data[csv_col]
        
        supabase.table("deals").update(db_data).eq("id", deal_id).execute()
        return True
    except Exception as e:
        st.error(f"Supabase update error: {e}")
        return False


def delete_deal_from_supabase(deal_id: int) -> bool:
    """Delete a deal from Supabase."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        supabase.table("deals").delete().eq("id", deal_id).execute()
        return True
    except Exception as e:
        st.error(f"Supabase delete error: {e}")
        return False


def save_data_to_csv(df: pd.DataFrame):
    """Save DataFrame to CSV file."""
    df.to_csv(CSV_FILE, index=False)


def save_data(df: pd.DataFrame):
    """Save data (currently uses CSV as backup)."""
    save_data_to_csv(df)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_monetary_value(value):
    """Parse monetary string like '600M', '2.5B', '$14M' into numeric value in millions."""
    if pd.isna(value) or value == "N/A" or value == "" or value == "None":
        return 0
    
    value_str = str(value).upper().strip()
    value_str = value_str.replace("$", "").replace(",", "").strip()
    
    try:
        if "B" in value_str:
            num = float(value_str.replace("B", "").strip())
            return num * 1000
        elif "M" in value_str:
            num = float(value_str.replace("M", "").strip())
            return num
        elif "K" in value_str:
            num = float(value_str.replace("K", "").strip())
            return num / 1000
        else:
            return float(value_str)
    except:
        return 0


def get_exa_client():
    """Get Exa client with API key."""
    api_key = st.secrets.get("EXA_API_KEY") or os.getenv("EXA_API_KEY")
    if not api_key:
        raise Exception("EXA_API_KEY not found. Please set it in Streamlit secrets or environment variable.")
    return Exa(api_key=api_key)


def fetch_article_with_exa(url):
    """Fetch article content using Exa's get_contents API."""
    try:
        exa = get_exa_client()
        st.info("Fetching article content via Exa...")
        
        result = exa.get_contents([url], text=True)
        
        if result.results and len(result.results) > 0:
            article = result.results[0]
            text_content = getattr(article, 'text', '') or ''
            title = getattr(article, 'title', '') or ''
            
            if text_content:
                st.success(f"Fetched article: {title[:50]}...")
                return f"Title: {title}\n\nContent:\n{text_content[:8000]}"
        
        st.warning("Exa couldn't fetch article content, will search for information instead")
        return None
    except Exception as e:
        st.warning(f"Exa fetch failed: {e}")
        return None


def search_article_with_exa(url):
    """Search for article information using Exa when direct fetch fails."""
    try:
        exa = get_exa_client()
        st.info("Searching for article information via Exa...")
        
        results = exa.search_and_contents(
            url,
            num_results=3,
            text=True
        )
        
        if results.results:
            combined = []
            for r in results.results:
                text = getattr(r, 'text', '') or ''
                combined.append(f"Title: {r.title}\nURL: {r.url}\nContent: {text[:2000]}\n---")
            return "\n\n".join(combined)[:8000]
        
        return None
    except Exception as e:
        st.warning(f"Exa search failed: {e}")
        return None


def search_company_info(company_name):
    """Search for company information using Exa."""
    try:
        exa = get_exa_client()
        st.info(f"Searching for company info: {company_name}")
        
        search_queries = [
            f"{company_name} company revenue annual 2024 2023",
            f"{company_name} funding round valuation Crunchbase",
            f"{company_name} Series funding raised total"
        ]
        
        all_results = []
        for query in search_queries:
            try:
                st.info(f"Searching: {query}")
                results = exa.search_and_contents(
                    query,
                    num_results=5,
                    text=True
                )
                st.info(f"Got {len(results.results)} results")
                for result in results.results:
                    text_content = getattr(result, 'text', '') or ''
                    all_results.append({
                        "title": result.title,
                        "url": result.url,
                        "text": text_content[:2000] if text_content else ""
                    })
            except Exception as e:
                st.warning(f"Query failed: {e}")
                continue
        
        if all_results:
            st.success(f"Found {len(all_results)} results for {company_name}")
        else:
            st.warning(f"No results found for {company_name}")
        
        combined_text = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['text']}\n---"
            for r in all_results[:10]
        ])
        
        return combined_text[:10000] if combined_text else ""
    except Exception as e:
        st.error(f"Search error: {e}")
        return ""


def extract_deal_info(text, company_name=None):
    """Extract deal information using OpenAI with web search enhancement."""
    
    st.info(f"Article text length: {len(text)} characters")
    if len(text) < 100:
        st.error(f"Article text is too short! Content: {text[:500]}")
    
    initial_prompt = """Extract the following information from this M&A news article and return ONLY a valid JSON object with these exact keys:
- "Company (Target)": The target company being acquired
- "Acquirer": The acquiring company
- "Category": The market/industry category
- "Amount": Deal value in millions or billions (e.g., "500M", "2.5B", or "N/A" if not available)
- "Rationale": One sentence summary of why the deal happened

If any information is missing, use "N/A" for that field. Return ONLY the JSON object, no other text.

Article text:
""" + text

    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OpenAI API key not found. Please set it in Streamlit secrets or as OPENAI_API_KEY environment variable.")
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": initial_prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        basic_data = json.loads(content)
        st.info(f"Initial extraction: {basic_data}")
        extracted_company_name = basic_data.get("Company (Target)", company_name or "")
        
        company_info_text = ""
        if extracted_company_name and extracted_company_name != "N/A":
            with st.spinner(f"Searching web for {extracted_company_name}..."):
                company_info_text = search_company_info(extracted_company_name)
        
        enhanced_prompt = """You have two sources of information:
1. An M&A news article
2. Web search results about the target company

CRITICAL: You MUST search the web results carefully for revenue data. Look for:
- Annual revenue figures (2024, 2023, etc.)
- Revenue statements in financial reports
- Revenue mentioned in funding announcements
- Revenue data from Crunchbase, company websites, or news articles

Extract and combine information from BOTH sources to return ONLY a valid JSON object with these exact keys:
- "Company (Target)": The target company being acquired
- "Acquirer": The acquiring company
- "Category": The market/industry category
- "Amount": Deal value in millions or billions (e.g., "500M", "2.5B", or "N/A" if not available)
- "Revenue": Target company's revenue - SEARCH THE WEB RESULTS CAREFULLY for this! Look for "$X million", "$X billion", "revenue of $X", etc. If found, format as "14M" or "100M" etc. If truly not found, use "N/A"
- "Multiple": EV/Revenue multiple - YOU MUST CALCULATE THIS if both Amount and Revenue are available: (Deal Amount in millions) / (Revenue in millions). Format as "464x" or "25.5x" etc. If cannot calculate, use "N/A"
- "Rationale": One sentence summary of why the deal happened
- "Company URL": The target company's website URL (from web search or article), else "N/A"
- "Company Description": A short description of what the target company does (1-2 sentences max), else "N/A"
- "Last Round Raised": The amount raised in the most recent funding round (e.g., "50M", "100M", or "N/A" if not available)
- "Valuation of Last Round": The valuation from the most recent funding round (e.g., "500M", "1B", or "N/A" if not available)
- "Total Raised": Total funding raised by the target company across all rounds (e.g., "200M", "500M", or "N/A" if not available)

CALCULATION RULES:
- Deal Amount "2.5B" = 2500M, "500M" = 500M
- Revenue "$14 million" = 14M, "$100M" = 100M, "$1.2 billion" = 1200M
- Multiple = (Deal Amount in M) / (Revenue in M)
- Example: Deal $6.5B, Revenue $14M → Multiple = 6500 / 14 = 464x

Return ONLY the JSON object, no other text.

M&A Article text:
""" + text + """

Web search results about the company (SEARCH THESE CAREFULLY FOR REVENUE DATA):
""" + company_info_text

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant. Use web search results to find missing information. Return only valid JSON."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        required_fields = [
            "Company (Target)", "Acquirer", "Category", "Amount", "Revenue", "Multiple", "Rationale",
            "Company URL", "Company Description", "Last Round Raised", "Valuation of Last Round", "Total Raised"
        ]
        for field in required_fields:
            if field not in data:
                data[field] = "N/A"
        
        # Recalculate multiple
        try:
            amount_str = str(data.get("Amount", "")).upper().strip()
            revenue_str = str(data.get("Revenue", "")).upper().strip()
            
            if amount_str != "N/A" and revenue_str != "N/A" and amount_str and revenue_str:
                amount_val = None
                amount_clean = amount_str.replace("$", "").replace(",", "").replace(" ", "").strip()
                if "B" in amount_clean or "BILLION" in amount_clean:
                    num_part = amount_clean.replace("B", "").replace("BILLION", "").strip()
                    amount_val = float(num_part) * 1000
                elif "M" in amount_clean or "MILLION" in amount_clean:
                    num_part = amount_clean.replace("M", "").replace("MILLION", "").strip()
                    amount_val = float(num_part)
                elif amount_clean.replace(".", "").isdigit():
                    amount_val = float(amount_clean)
                
                revenue_val = None
                revenue_clean = revenue_str.replace("$", "").replace(",", "").replace(" ", "").strip()
                if "B" in revenue_clean or "BILLION" in revenue_clean:
                    num_part = revenue_clean.replace("B", "").replace("BILLION", "").strip()
                    revenue_val = float(num_part) * 1000
                elif "M" in revenue_clean or "MILLION" in revenue_clean:
                    num_part = revenue_clean.replace("M", "").replace("MILLION", "").strip()
                    revenue_val = float(num_part)
                elif revenue_clean.replace(".", "").isdigit():
                    revenue_val = float(revenue_clean)
                
                if amount_val and revenue_val and revenue_val > 0:
                    multiple_val = amount_val / revenue_val
                    data["Multiple"] = f"{multiple_val:.0f}x" if multiple_val >= 100 else f"{multiple_val:.2f}x"
        except Exception as e:
            if data.get("Multiple") == "N/A" or not data.get("Multiple"):
                pass
        
        return data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main_app():
    """Main application UI (only shown to authenticated users)."""
    
    # Header with user info and logout
    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.markdown("""
        <h1>M&A_TRACKER_V1.0_<span class="blink">█</span></h1>
        <p class="byline">pace capital</p>
        """, unsafe_allow_html=True)
    with header_col2:
        st.markdown("""
        <style>
            /* Style the popover trigger */
            [data-testid="stPopover"] > div:first-child button {
                background-color: transparent !important;
                border: 1px solid #00FF00 !important;
                color: #00FF00 !important;
                font-family: 'Geist Mono', 'SF Mono', 'Consolas', monospace !important;
                font-size: 12px !important;
                padding: 6px 16px !important;
                border-radius: 0 !important;
                white-space: nowrap !important;
            }
            [data-testid="stPopover"] > div:first-child button:hover {
                background-color: #00FF00 !important;
                color: #000000 !important;
            }
            /* Push popover to the right */
            [data-testid="stPopover"] {
                float: right;
            }
        </style>
        """, unsafe_allow_html=True)
        
        with st.popover(st.session_state.user_email):
            if st.button("[ LOGOUT ]", key="logout_btn", use_container_width=True):
                logout()
    
    # Data source indicator
    if is_supabase_configured():
        st.markdown('<span style="color: #00FF00; font-size: 10px;">● CONNECTED TO CLOUD DATABASE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #FFAA00; font-size: 10px;">● USING LOCAL CSV (Configure Supabase for cloud sync)</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    if 'df_cache' not in st.session_state or 'df_cache_timestamp' not in st.session_state:
        df = load_data()
        st.session_state.df_cache = df
        st.session_state.df_cache_timestamp = datetime.now().timestamp()
    else:
        # Refresh every 30 seconds or on manual refresh
        if datetime.now().timestamp() - st.session_state.df_cache_timestamp > 30:
            df = load_data()
            st.session_state.df_cache = df
            st.session_state.df_cache_timestamp = datetime.now().timestamp()
        else:
            df = st.session_state.df_cache.copy()
    
    # Input section
    st.markdown("### [ INPUT ]")
    url = st.text_input("Article URL:", placeholder="https://example.com/news/acquisition-article", key="url_input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ingest_button = st.button("[ INGEST ]", key="ingest_btn")
    
    if ingest_button and url:
        if not url.startswith(("http://", "https://")):
            st.markdown('<div class="error-message">ERROR: Invalid URL format</div>', unsafe_allow_html=True)
        else:
            st.session_state.processing_ingestion = True
            
            with st.spinner("Fetching article via Exa..."):
                try:
                    article_text = fetch_article_with_exa(url)
                    
                    if not article_text:
                        article_text = search_article_with_exa(url)
                    
                    if article_text:
                        st.markdown('<div class="success-message">✓ Article fetched via Exa</div>', unsafe_allow_html=True)
                        
                        with st.expander("Debug: Fetched article content (first 1000 chars)", expanded=False):
                            st.text(article_text[:1000] if article_text else "NO CONTENT")
                        
                        with st.spinner("Extracting deal information..."):
                            deal_data = extract_deal_info(article_text)
                            
                            if deal_data:
                                deal_data["Date Added"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Try to save to Supabase first
                                if is_supabase_configured():
                                    if save_deal_to_supabase(deal_data):
                                        st.markdown('<div class="success-message">✓ Data saved to cloud database</div>', unsafe_allow_html=True)
                                    else:
                                        st.warning("Failed to save to cloud. Saving to local CSV...")
                                        new_row = pd.DataFrame([deal_data])
                                        df = pd.concat([df, new_row], ignore_index=True)
                                        save_data(df)
                                else:
                                    new_row = pd.DataFrame([deal_data])
                                    df = pd.concat([df, new_row], ignore_index=True)
                                    save_data(df)
                                    st.markdown('<div class="success-message">✓ Data saved to local CSV</div>', unsafe_allow_html=True)
                                
                                # Refresh cache
                                st.session_state.df_cache = load_data()
                                st.session_state.df_cache_timestamp = datetime.now().timestamp()
                                df = st.session_state.df_cache.copy()
                                
                                st.json(deal_data)
                                
                                if 'last_editor_hash' in st.session_state:
                                    del st.session_state.last_editor_hash
                            else:
                                st.markdown('<div class="error-message">ERROR: Failed to extract deal information</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">ERROR: Could not fetch article content via Exa</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-message">ERROR: {str(e)}</div>', unsafe_allow_html=True)
                finally:
                    st.session_state.processing_ingestion = False
    
    st.markdown("---")
    
    # Data table with scalable UI
    if not df.empty:
        st.markdown("### [ DATABASE ]")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search:", placeholder="Search by company, acquirer, category...", key="search_input")
        with col2:
            rows_per_page = st.selectbox("Rows per page:", options=[25, 50, 100, 200], index=0, key="rows_per_page")
        with col3:
            st.write("")
            st.write(f"**Total entries:** {len(df)}")
        
        if search_query:
            search_lower = search_query.lower()
            mask = (
                df["Company (Target)"].astype(str).str.lower().str.contains(search_lower, na=False) |
                df["Acquirer"].astype(str).str.lower().str.contains(search_lower, na=False) |
                df["Category"].astype(str).str.lower().str.contains(search_lower, na=False) |
                df["Amount"].astype(str).str.lower().str.contains(search_lower, na=False)
            )
            filtered_df = df[mask].copy()
        else:
            filtered_df = df
        
        total_pages = max(1, (len(filtered_df) + rows_per_page - 1) // rows_per_page)
        if total_pages > 1:
            page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, key="page_num")
            start_idx = (page_num - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            paginated_df = filtered_df.iloc[start_idx:end_idx].copy()
            st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(filtered_df))} of {len(filtered_df)} entries")
        else:
            paginated_df = filtered_df.copy()
            page_num = 1
            start_idx = 0
            end_idx = len(filtered_df)
        
        if not paginated_df.empty:
            display_df = paginated_df.copy()
            display_df.insert(0, '#', range(start_idx + 1 if total_pages > 1 else 1, len(display_df) + (start_idx + 1 if total_pages > 1 else 1)))
            
            display_df['Amount'] = display_df['Amount'].apply(parse_monetary_value)
            display_df['Revenue'] = display_df['Revenue'].apply(parse_monetary_value)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Deal value in millions",
                        format="%.0fM"
                    ),
                    "Revenue": st.column_config.NumberColumn(
                        "Revenue",
                        help="Revenue in millions",
                        format="%.1fM"
                    ),
                }
            )
            
            if 'edit_mode' not in st.session_state:
                st.session_state.edit_mode = False
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("[ EDIT MODE ]" if not st.session_state.edit_mode else "[ VIEW MODE ]", key="toggle_edit"):
                    st.session_state.edit_mode = not st.session_state.edit_mode
                    st.rerun()
            
            if st.session_state.edit_mode:
                edit_options = []
                for i, (idx, row) in enumerate(paginated_df.iterrows()):
                    row_num = start_idx + i + 1
                    edit_options.append((i, idx, row_num, f"#{row_num}: {row['Company (Target)']} <- {row['Acquirer']}"))
                
                selected_option = st.selectbox(
                    "Select entry to edit:",
                    options=range(len(edit_options)),
                    format_func=lambda x: edit_options[x][3],
                    key="edit_selector"
                )
                
                if selected_option is not None:
                    i, idx, row_num, _ = edit_options[selected_option]
                    row = paginated_df.iloc[i]
                    
                    st.markdown("---")
                    
                    with st.form(f"edit_form_{row_num}", clear_on_submit=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            edit_target = st.text_input("Company (Target):", value=str(row.get("Company (Target)", "")), key=f"target_{row_num}")
                            edit_acquirer = st.text_input("Acquirer:", value=str(row.get("Acquirer", "")), key=f"acquirer_{row_num}")
                            edit_category = st.text_input("Category:", value=str(row.get("Category", "")), key=f"category_{row_num}")
                            edit_amount = st.text_input("Amount:", value=str(row.get("Amount", "")), key=f"amount_{row_num}")
                            edit_revenue = st.text_input("Revenue:", value=str(row.get("Revenue", "")), key=f"revenue_{row_num}")
                            edit_url = st.text_input("Company URL:", value=str(row.get("Company URL", "")), key=f"url_{row_num}")
                        
                        with col2:
                            edit_last_round = st.text_input("Last Round Raised:", value=str(row.get("Last Round Raised", "")), key=f"last_{row_num}")
                            edit_valuation = st.text_input("Valuation of Last Round:", value=str(row.get("Valuation of Last Round", "")), key=f"val_{row_num}")
                            edit_total = st.text_input("Total Raised:", value=str(row.get("Total Raised", "")), key=f"total_{row_num}")
                            edit_desc = st.text_area("Company Description:", value=str(row.get("Company Description", "")), height=80, key=f"desc_{row_num}")
                            edit_rationale = st.text_area("Rationale:", value=str(row.get("Rationale", "")), height=80, key=f"rationale_{row_num}")
                        
                        calculated_multiple = str(row.get("Multiple", "N/A"))
                        if edit_amount and edit_revenue and str(edit_amount).upper().strip() != "N/A" and str(edit_revenue).upper().strip() != "N/A":
                            try:
                                amount_str = str(edit_amount).upper().strip().replace("$", "").replace(",", "").replace(" ", "")
                                revenue_str = str(edit_revenue).upper().strip().replace("$", "").replace(",", "").replace(" ", "")
                                
                                amount_val = None
                                if "B" in amount_str:
                                    amount_val = float(amount_str.replace("B", "").strip()) * 1000
                                elif "M" in amount_str:
                                    amount_val = float(amount_str.replace("M", "").strip())
                                
                                revenue_val = None
                                if "B" in revenue_str:
                                    revenue_val = float(revenue_str.replace("B", "").strip()) * 1000
                                elif "M" in revenue_str:
                                    revenue_val = float(revenue_str.replace("M", "").strip())
                                
                                if amount_val and revenue_val and revenue_val > 0:
                                    multiple_val = amount_val / revenue_val
                                    calculated_multiple = f"{multiple_val:.0f}x" if multiple_val >= 100 else f"{multiple_val:.2f}x"
                            except:
                                pass
                        
                        st.markdown(f"**Multiple (auto-calculated):** {calculated_multiple}")
                        
                        col_save, col_delete = st.columns([1, 1])
                        with col_save:
                            save_btn = st.form_submit_button("[ SAVE CHANGES ]")
                        with col_delete:
                            delete_btn = st.form_submit_button("[ DELETE ENTRY ]")
                        
                        if delete_btn:
                            df = df.drop(idx).reset_index(drop=True)
                            save_data(df)
                            st.session_state.df_cache = df
                            st.session_state.df_cache_timestamp = datetime.now().timestamp()
                            st.session_state.edit_mode = False
                            st.success(f"Entry #{row_num} deleted!")
                            st.rerun()
                        
                        if save_btn:
                            df.at[idx, "Company (Target)"] = edit_target.strip() if edit_target else "N/A"
                            df.at[idx, "Acquirer"] = edit_acquirer.strip() if edit_acquirer else "N/A"
                            df.at[idx, "Category"] = edit_category.strip() if edit_category else "N/A"
                            df.at[idx, "Amount"] = edit_amount.strip() if edit_amount else "N/A"
                            df.at[idx, "Revenue"] = edit_revenue.strip() if edit_revenue else "N/A"
                            df.at[idx, "Multiple"] = calculated_multiple
                            df.at[idx, "Rationale"] = edit_rationale.strip() if edit_rationale else "N/A"
                            df.at[idx, "Company URL"] = edit_url.strip() if edit_url else "N/A"
                            df.at[idx, "Company Description"] = edit_desc.strip() if edit_desc else "N/A"
                            df.at[idx, "Last Round Raised"] = edit_last_round.strip() if edit_last_round else "N/A"
                            df.at[idx, "Valuation of Last Round"] = edit_valuation.strip() if edit_valuation else "N/A"
                            df.at[idx, "Total Raised"] = edit_total.strip() if edit_total else "N/A"
                            
                            save_data(df)
                            st.session_state.df_cache = df
                            st.session_state.df_cache_timestamp = datetime.now().timestamp()
                            st.success(f"Entry #{row_num} saved!")
                            st.rerun()
    
    else:
        st.markdown("### [ DATABASE ]")
        st.markdown("Database empty. Ingest your first deal to begin tracking.")


# ============================================================================
# ENTRY POINT
# ============================================================================

# Initialize authentication
init_auth()

# Show appropriate page based on auth state
if st.session_state.authenticated:
    main_app()
else:
    show_login_page()
