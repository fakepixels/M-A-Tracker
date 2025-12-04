import streamlit as st
import pandas as pd
from playwright.sync_api import sync_playwright
import openai
import os
import json
from datetime import datetime
from duckduckgo_search import DDGS

# Try to import Exa for better search
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False


# Configure page
st.set_page_config(
    page_title="M&A Tracker",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Terminal-style CSS
st.markdown("""
<style>
    @import url('https://geistfont.vercel.app/geist.css');
    
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .main .block-container {
        background-color: #000000;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6, p, div, span, label, input, textarea, button {
        font-family: 'Geist Mono', monospace !important;
        color: #FFFFFF !important;
    }
    
    h1 {
        color: #888888 !important;
        font-weight: bold;
        letter-spacing: 2px;
        margin-bottom: 0 !important;
    }
    
    .byline {
        color: #888888;
        font-family: 'Geist Mono', monospace;
        font-size: 0.8rem;
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
        border: 1px solid #FFFFFF;
        color: #FFFFFF;
        border-radius: 0px;
        font-family: 'Geist Mono', monospace !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid #00FF00;
        box-shadow: 0 0 5px #00FF00;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #000000;
        border: 1px solid #FFFFFF;
        color: #FFFFFF;
        border-radius: 0px;
        font-family: 'Geist Mono', monospace !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 1px solid #00FF00;
        box-shadow: 0 0 5px #00FF00;
    }
    
    button {
        background-color: #000000 !important;
        border: 1px solid #FFFFFF !important;
        color: #FFFFFF !important;
        border-radius: 0px !important;
        font-family: 'Geist Mono', monospace !important;
        padding: 0.5rem 1rem !important;
    }
    
    button:hover {
        border-color: #00FF00 !important;
        color: #00FF00 !important;
        box-shadow: 0 0 5px #00FF00 !important;
    }
    
    .stDataFrame {
        background-color: #000000;
        border: 1px solid #FFFFFF;
    }
    
    .metric-container {
        border: 1px solid #FFFFFF;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #000000;
    }
    
    .success-message {
        color: #00FF00;
        font-family: 'Geist Mono', monospace;
        border: 1px solid #00FF00;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .error-message {
        color: #FF0000;
        font-family: 'Geist Mono', monospace;
        border: 1px solid #FF0000;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-message {
        color: #FFAA00;
        font-family: 'Geist Mono', monospace;
        border: 1px solid #FFAA00;
        padding: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# CSV file path
CSV_FILE = "mergers.csv"

def load_data():
    """Load data from CSV file, create if it doesn't exist."""
    # Define all required columns
    required_columns = [
        "Company (Target)", "Acquirer", "Category", "Amount", "Revenue", "Multiple", "Rationale",
        "Company URL", "Company Description", "Last Round Raised", "Valuation of Last Round", "Total Raised",
        "Date Added"
    ]
    
    if os.path.exists(CSV_FILE):
        try:
            # Check if file is empty
            if os.path.getsize(CSV_FILE) == 0:
                return pd.DataFrame(columns=required_columns)
            
            df = pd.read_csv(CSV_FILE)
            
            # Check if dataframe is empty
            if df.empty:
                return pd.DataFrame(columns=required_columns)
            
            # Add any missing columns (for backward compatibility)
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "N/A"
            return df
        except pd.errors.EmptyDataError:
            # File exists but is empty or has no valid data
            return pd.DataFrame(columns=required_columns)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(columns=required_columns)
    else:
        # Create empty DataFrame with required columns
        return pd.DataFrame(columns=required_columns)

def save_data(df):
    """Save DataFrame to CSV file."""
    df.to_csv(CSV_FILE, index=False)

def scrape_article(url):
    """Scrape text content from a URL using Playwright for advanced scraping."""
    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            # Navigate to URL and wait for content to load
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait a bit for any dynamic content
            page.wait_for_timeout(2000)
            
            # Try to find main article content using common selectors
            # This handles paywalls, dynamic loading, etc.
            article_selectors = [
                'article',
                '[role="article"]',
                '.article-content',
                '.article-body',
                '.post-content',
                '.entry-content',
                'main',
                '.content',
                'body'
            ]
            
            text_content = None
            for selector in article_selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        text_content = element.inner_text()
                        if len(text_content) > 500:  # Found substantial content
                            break
                except:
                    continue
            
            # Fallback to body if no article found
            if not text_content or len(text_content) < 500:
                text_content = page.inner_text('body')
            
            browser.close()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:8000]  # Limit to 8000 characters for API
    except Exception as e:
        raise Exception(f"Error scraping URL: {str(e)}")

def search_with_exa(company_name):
    """Search using Exa API for better company information."""
    try:
        exa_api_key = st.secrets.get("EXA_API_KEY") or os.getenv("EXA_API_KEY")
        if not exa_api_key:
            st.warning("⚠️ EXA_API_KEY not found in secrets")
            return None
        
        st.info(f"🔍 Using Exa Search for: {company_name}")
        exa = Exa(api_key=exa_api_key)
        
        # Search for company revenue and funding information
        search_queries = [
            f"{company_name} company revenue annual 2024 2023",
            f"{company_name} funding round valuation Crunchbase",
            f"{company_name} Series funding raised total"
        ]
        
        all_results = []
        for query in search_queries:
            try:
                st.info(f"  → Searching: {query}")
                # Use correct Exa API parameters
                results = exa.search_and_contents(
                    query,
                    num_results=5,
                    text=True  # Get full text content
                )
                st.info(f"  → Got {len(results.results)} results for this query")
                for result in results.results:
                    text_content = getattr(result, 'text', '') or ''
                    all_results.append({
                        "title": result.title,
                        "url": result.url,
                        "text": text_content[:2000] if text_content else ""
                    })
            except Exception as e:
                st.warning(f"Exa query failed: {e}")
                continue
        
        if all_results:
            st.success(f"✓ Exa found {len(all_results)} total results")
        else:
            st.warning("Exa returned no results")
        
        # Combine results
        combined_text = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['text']}\n---"
            for r in all_results[:10]
        ])
        
        return combined_text[:10000] if combined_text else None
    except Exception as e:
        st.error(f"Exa error: {e}")
        return None

def search_company_info(company_name):
    """Search the web for company information with targeted queries."""
    # Try Exa first if available
    if EXA_AVAILABLE:
        st.info("Exa library is available, attempting Exa search...")
        exa_results = search_with_exa(company_name)
        if exa_results:
            return exa_results
        st.warning("Exa search failed, falling back to DuckDuckGo...")
    else:
        st.warning("Exa library not available, using DuckDuckGo...")
    
    # Fall back to DuckDuckGo
    try:
        with DDGS() as ddgs:
            # More specific search queries for better results
            search_queries = [
                f"{company_name} revenue 2024 2023 annual ARR",
                f"{company_name} funding round Series valuation Crunchbase",
                f"{company_name} raised million funding",
                f"{company_name} company revenue financials",
                f'site:crunchbase.com "{company_name}"',
                f'site:pitchbook.com "{company_name}"'
            ]
            
            search_results = []
            for query in search_queries:
                try:
                    results = list(ddgs.text(query, max_results=5))
                    search_results.extend(results)
                except Exception as e:
                    continue
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            for r in search_results:
                url = r.get('href', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(r)
            
            # Combine all search result snippets with more detail
            combined_text = "\n\n".join([
                f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nContent: {r.get('body', '')}\n---" 
                for r in unique_results[:15]  # Get more results
            ])
            
            return combined_text[:8000]  # Increase limit for more context
    except Exception as e:
        return f"Error searching: {str(e)}"

def extract_deal_info(text, company_name=None):
    """Extract deal information using OpenAI with web search enhancement."""
    
    # Debug: Show article text length
    st.info(f"📄 Article text length: {len(text)} characters")
    if len(text) < 100:
        st.error(f"Article text is too short! Content: {text[:500]}")
    
    # First, extract basic deal info from the article
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
        # Try Streamlit secrets first, then fall back to environment variable
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OpenAI API key not found. Please set it in Streamlit secrets or as OPENAI_API_KEY environment variable.")
        client = openai.OpenAI(api_key=api_key)
        
        # First extraction: get basic deal info
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": initial_prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        basic_data = json.loads(content)
        st.info(f"📊 Initial extraction: {basic_data}")
        extracted_company_name = basic_data.get("Company (Target)", company_name or "")
        
        # Now search the web for additional company information
        company_info_text = ""
        if extracted_company_name and extracted_company_name != "N/A":
            with st.spinner(f"Searching web for {extracted_company_name}..."):
                company_info_text = search_company_info(extracted_company_name)
        
        # Enhanced extraction with web search results
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

        # Second extraction: enhanced with web search
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant. Use web search results to find missing information. Return only valid JSON."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        # Ensure all required fields are present
        required_fields = [
            "Company (Target)", "Acquirer", "Category", "Amount", "Revenue", "Multiple", "Rationale",
            "Company URL", "Company Description", "Last Round Raised", "Valuation of Last Round", "Total Raised"
        ]
        for field in required_fields:
            if field not in data:
                data[field] = "N/A"
        
        # Always recalculate multiple if we have both Amount and Revenue
        try:
            amount_str = str(data.get("Amount", "")).upper().strip()
            revenue_str = str(data.get("Revenue", "")).upper().strip()
            
            if amount_str != "N/A" and revenue_str != "N/A" and amount_str and revenue_str:
                # Parse amount - handle various formats
                amount_val = None
                amount_clean = amount_str.replace("$", "").replace(",", "").replace(" ", "").strip()
                if "B" in amount_clean or "BILLION" in amount_clean:
                    num_part = amount_clean.replace("B", "").replace("BILLION", "").strip()
                    amount_val = float(num_part) * 1000
                elif "M" in amount_clean or "MILLION" in amount_clean:
                    num_part = amount_clean.replace("M", "").replace("MILLION", "").strip()
                    amount_val = float(num_part)
                elif amount_clean.replace(".", "").isdigit():
                    # Assume millions if just a number
                    amount_val = float(amount_clean)
                
                # Parse revenue - handle various formats
                revenue_val = None
                revenue_clean = revenue_str.replace("$", "").replace(",", "").replace(" ", "").strip()
                if "B" in revenue_clean or "BILLION" in revenue_clean:
                    num_part = revenue_clean.replace("B", "").replace("BILLION", "").strip()
                    revenue_val = float(num_part) * 1000
                elif "M" in revenue_clean or "MILLION" in revenue_clean:
                    num_part = revenue_clean.replace("M", "").replace("MILLION", "").strip()
                    revenue_val = float(num_part)
                elif revenue_clean.replace(".", "").isdigit():
                    # Assume millions if just a number
                    revenue_val = float(revenue_clean)
                
                # Calculate multiple if we have both values
                if amount_val and revenue_val and revenue_val > 0:
                    multiple_val = amount_val / revenue_val
                    data["Multiple"] = f"{multiple_val:.0f}x" if multiple_val >= 100 else f"{multiple_val:.2f}x"
        except Exception as e:
            # If calculation fails, keep whatever the AI provided or "N/A"
            if data.get("Multiple") == "N/A" or not data.get("Multiple"):
                pass  # Keep as N/A
        
        return data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

# Main app
st.markdown("""
<h1>M&A_TRACKER_V1.0_<span class="blink">█</span></h1>
<p class="byline">made by tina</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Load existing data with caching
if 'df_cache' not in st.session_state or 'df_cache_timestamp' not in st.session_state:
    df = load_data()
    st.session_state.df_cache = df
    st.session_state.df_cache_timestamp = os.path.getmtime(CSV_FILE) if os.path.exists(CSV_FILE) else 0
else:
    # Check if CSV file has been modified
    current_timestamp = os.path.getmtime(CSV_FILE) if os.path.exists(CSV_FILE) else 0
    if current_timestamp > st.session_state.df_cache_timestamp:
        df = load_data()
        st.session_state.df_cache = df
        st.session_state.df_cache_timestamp = current_timestamp
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
        # Set flag to prevent data editor from processing during ingestion
        st.session_state.processing_ingestion = True
        
        with st.spinner("Scraping article..."):
            try:
                article_text = scrape_article(url)
                st.markdown('<div class="success-message">✓ Article scraped successfully</div>', unsafe_allow_html=True)
                
                # Debug: Show first 500 chars of scraped content
                with st.expander("🔍 Debug: Scraped article content (first 1000 chars)", expanded=False):
                    st.text(article_text[:1000] if article_text else "NO CONTENT")
                
                with st.spinner("Extracting deal information..."):
                    deal_data = extract_deal_info(article_text)
                    
                    if deal_data:
                        # Add date
                        deal_data["Date Added"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Convert to DataFrame row
                        new_row = pd.DataFrame([deal_data])
                        
                        # Append to existing data
                        df = pd.concat([df, new_row], ignore_index=True)
                        
                        # Save to CSV
                        save_data(df)
                        
                        # Update cache
                        st.session_state.df_cache = df
                        st.session_state.df_cache_timestamp = os.path.getmtime(CSV_FILE) if os.path.exists(CSV_FILE) else 0
                        
                        st.markdown('<div class="success-message">✓ Data ingested and saved to mergers.csv</div>', unsafe_allow_html=True)
                        st.json(deal_data)
                        
                        # Clear the editor hash to force refresh
                        if 'last_editor_hash' in st.session_state:
                            del st.session_state.last_editor_hash
                    else:
                        st.markdown('<div class="error-message">ERROR: Failed to extract deal information</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">ERROR: {str(e)}</div>', unsafe_allow_html=True)
            finally:
                st.session_state.processing_ingestion = False
                # Don't rerun immediately - let user see the result first

st.markdown("---")

# Data table with scalable UI
if not df.empty:
    st.markdown("### [ DATABASE ]")
    
    # Search and filter section
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search:", placeholder="Search by company, acquirer, category...", key="search_input")
    with col2:
        rows_per_page = st.selectbox("Rows per page:", options=[25, 50, 100, 200], index=0, key="rows_per_page")
    with col3:
        st.write("")  # Spacing
        st.write(f"**Total entries:** {len(df)}")
    
    # Filter dataframe based on search (optimized)
    if search_query:
        search_lower = search_query.lower()
        # Use vectorized operations for better performance
        mask = (
            df["Company (Target)"].astype(str).str.lower().str.contains(search_lower, na=False) |
            df["Acquirer"].astype(str).str.lower().str.contains(search_lower, na=False) |
            df["Category"].astype(str).str.lower().str.contains(search_lower, na=False) |
            df["Amount"].astype(str).str.lower().str.contains(search_lower, na=False)
        )
        filtered_df = df[mask].copy()
    else:
        filtered_df = df
    
    # Pagination
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
    
    # Use data_editor for inline editing
    if not paginated_df.empty:
        # Add row numbers
        display_df = paginated_df.copy()
        display_df.insert(0, '#', range(start_idx + 1 if total_pages > 1 else 1, len(display_df) + (start_idx + 1 if total_pages > 1 else 1)))
        
        # Configure editable columns (exclude # and Date Added)
        editable_columns = {
            "Company (Target)": True,
            "Acquirer": True,
            "Category": True,
            "Amount": True,
            "Revenue": True,
            "Multiple": False,  # Auto-calculated, not editable
            "Rationale": True,
            "Company URL": True,
            "Company Description": True,
            "Last Round Raised": True,
            "Valuation of Last Round": True,
            "Total Raised": True,
        }
        
        # Use regular dataframe display - editing will be done via a separate edit button
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Edit button to enable editing mode
        if 'edit_mode' not in st.session_state:
            st.session_state.edit_mode = False
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("[ EDIT MODE ]" if not st.session_state.edit_mode else "[ VIEW MODE ]", key="toggle_edit"):
                st.session_state.edit_mode = not st.session_state.edit_mode
                st.rerun()
        
        # Show editable form when in edit mode
        if st.session_state.edit_mode:
            # Create selection options
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
                    
                    # Show calculated multiple
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
                        # Map back to original index
                        if search_query or total_pages > 1:
                            orig_idx = filtered_df.index[start_idx + i]
                        else:
                            orig_idx = idx
                        
                        # Delete the row
                        df = df.drop(df.index[orig_idx]).reset_index(drop=True)
                        save_data(df)
                        st.session_state.df_cache = df
                        st.session_state.df_cache_timestamp = os.path.getmtime(CSV_FILE) if os.path.exists(CSV_FILE) else 0
                        st.success(f"Entry #{row_num} deleted!")
                        st.rerun()
                    
                    if save_btn:
                        # Map back to original index
                        if search_query or total_pages > 1:
                            orig_idx = filtered_df.index[start_idx + i]
                        else:
                            orig_idx = idx
                        
                        # Update row
                        df.at[orig_idx, "Company (Target)"] = edit_target.strip() if edit_target else "N/A"
                        df.at[orig_idx, "Acquirer"] = edit_acquirer.strip() if edit_acquirer else "N/A"
                        df.at[orig_idx, "Category"] = edit_category.strip() if edit_category else "N/A"
                        df.at[orig_idx, "Amount"] = edit_amount.strip() if edit_amount else "N/A"
                        df.at[orig_idx, "Revenue"] = edit_revenue.strip() if edit_revenue else "N/A"
                        df.at[orig_idx, "Multiple"] = calculated_multiple
                        df.at[orig_idx, "Rationale"] = edit_rationale.strip() if edit_rationale else "N/A"
                        df.at[orig_idx, "Company URL"] = edit_url.strip() if edit_url else "N/A"
                        df.at[orig_idx, "Company Description"] = edit_desc.strip() if edit_desc else "N/A"
                        df.at[orig_idx, "Last Round Raised"] = edit_last_round.strip() if edit_last_round else "N/A"
                        df.at[orig_idx, "Valuation of Last Round"] = edit_valuation.strip() if edit_valuation else "N/A"
                        df.at[orig_idx, "Total Raised"] = edit_total.strip() if edit_total else "N/A"
                        
                        save_data(df)
                        st.session_state.df_cache = df
                        st.session_state.df_cache_timestamp = os.path.getmtime(CSV_FILE) if os.path.exists(CSV_FILE) else 0
                        st.success(f"Entry #{row_num} saved!")
                        st.rerun()

else:
    st.markdown("### [ DATABASE ]")
    st.markdown("Database empty. Ingest your first deal to begin tracking.")

