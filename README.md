# DealFlow Terminal

A terminal-style M&A tracking dashboard built with Streamlit. Track acquisitions, extract deal information from news articles, and manage your deal database with a sleek 90s hacker aesthetic.

## Features

- **URL Ingestion**: Paste news article URLs to automatically extract deal information using AI
- **Web Search Enhancement**: Automatically searches the web for company revenue, funding rounds, and valuation data
- **Manual Entry**: Add deals manually with a comprehensive form
- **Inline Editing**: Edit entries directly in the table with automatic multiple calculation
- **Search & Filter**: Quickly find deals by company, acquirer, or category
- **Pagination**: Handle hundreds of entries efficiently
- **Bulk Operations**: Delete multiple entries at once

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd sell-machine
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

4. Set up your OpenAI API key:
   - Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

5. Run the app:
```bash
streamlit run dealflow_terminal.py
```

## Usage

1. **Ingest from URL**: Paste a news article URL about an M&A deal and click `[ INGEST ]`
2. **Manual Entry**: Fill out the manual entry form to add deals directly
3. **Edit Entries**: Click on any cell in the table to edit inline
4. **Search**: Use the search bar to filter entries
5. **Delete**: Select row numbers and click `[ DELETE SELECTED ]`

## Data Fields

- Company (Target)
- Acquirer
- Category
- Amount (Deal value)
- Revenue
- Multiple (EV/Revenue - auto-calculated)
- Rationale
- Company URL
- Company Description
- Last Round Raised
- Valuation of Last Round
- Total Raised
- Date Added

## Tech Stack

- Streamlit
- OpenAI GPT-4o
- Playwright (web scraping)
- DuckDuckGo Search (web search)
- Pandas (data management)

## License

Private repository
