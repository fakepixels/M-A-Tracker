# M&A Tracker

A terminal-style M&A deal tracking dashboard built with Streamlit. Ingest acquisition news articles and automatically extract deal information using AI.

## Features

- 🔐 **Email-based Access Control** - Restricted to `@pacecapital.com` emails
- ☁️ **Cloud Database** - Data stored in Supabase (PostgreSQL)
- 🤖 **AI-Powered Extraction** - Automatically extract deal details from news articles
- 🔍 **Web Search Enhancement** - Uses Exa to fetch additional company information
- 📊 **Search & Filter** - Easily search and paginate through deals
- ✏️ **Inline Editing** - Edit deal information directly in the dashboard

## Quick Start

### 1. Clone and Install

```bash
cd sell-machine
pip install -r requirements.txt
```

### 2. Set Up Supabase (Cloud Database)

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to **SQL Editor** and run the contents of `supabase_setup.sql`
4. Go to **Settings > API** and copy your:
   - Project URL
   - `anon` public key
   - `service_role` key (for migration only)

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Supabase
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# OpenAI (for deal extraction)
OPENAI_API_KEY=sk-your-key

# Exa (for article fetching)
EXA_API_KEY=your-exa-key

# Auth
ALLOWED_EMAIL_DOMAIN=pacecapital.com
```

### 4. Migrate Existing Data (Optional)

If you have existing data in `mergers.csv`:

```bash
# Add SUPABASE_SERVICE_KEY to .env first
python migrate_csv_to_supabase.py
```

### 5. Run the App

```bash
streamlit run dealflow_terminal.py
```

## Deployment to Streamlit Cloud

### 1. Push to GitHub

```bash
git add .
git commit -m "Add cloud database and auth"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Select `dealflow_terminal.py` as the main file
4. Add your secrets in **Settings > Secrets**:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
OPENAI_API_KEY = "sk-..."
EXA_API_KEY = "..."
ALLOWED_EMAIL_DOMAIN = "pacecapital.com"
```

### 3. Share with Your Team

Share the Streamlit Cloud URL with your Pace Capital colleagues. Only users with `@pacecapital.com` emails can access the dashboard.

## File Structure

```
sell-machine/
├── dealflow_terminal.py    # Main Streamlit application
├── migrate_csv_to_supabase.py  # Migration script
├── supabase_setup.sql      # Database schema
├── requirements.txt        # Python dependencies
├── mergers.csv            # Local CSV backup
├── env_example.txt        # Environment variables template
└── README.md              # This file
```

## Security Notes

- The current auth uses email domain verification. For production, consider implementing proper Google OAuth with `streamlit-google-auth` or similar.
- Never commit `.env` files or expose API keys
- Supabase Row Level Security (RLS) is enabled for additional protection

## Upgrading to Full Google OAuth (Optional)

For enterprise-grade SSO, you can upgrade to full Google OAuth:

1. Create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com)
2. Install `streamlit-google-auth`:
   ```bash
   pip install streamlit-google-auth
   ```
3. Update the authentication section in `dealflow_terminal.py`

## Support

For issues or feature requests, contact the development team.

---

*Made with ❤️ by Pace Capital*
