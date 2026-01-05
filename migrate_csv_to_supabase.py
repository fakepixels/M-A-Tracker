"""
Migration script to import existing CSV data into Supabase.
Run this once after setting up your Supabase project.

Usage:
    python migrate_csv_to_supabase.py
"""

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CSV file path
CSV_FILE = "mergers.csv"

def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for migrations
    
    if not url or not key:
        raise Exception(
            "Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env file"
        )
    
    return create_client(url, key)


def migrate_csv_to_supabase():
    """Migrate data from CSV to Supabase."""
    
    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file not found: {CSV_FILE}")
        return
    
    # Load CSV data
    print(f"📄 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    if df.empty:
        print("⚠️  CSV file is empty, nothing to migrate.")
        return
    
    print(f"✅ Found {len(df)} records to migrate")
    
    # Connect to Supabase
    print("🔗 Connecting to Supabase...")
    supabase = get_supabase_client()
    
    # Prepare data for insertion
    # Map CSV columns to database columns
    column_mapping = {
        "Company (Target)": "company_target",
        "Acquirer": "acquirer",
        "Category": "category",
        "Amount": "amount",
        "Revenue": "revenue",
        "Multiple": "multiple",
        "Rationale": "rationale",
        "Company URL": "company_url",
        "Company Description": "company_description",
        "Last Round Raised": "last_round_raised",
        "Valuation of Last Round": "valuation_of_last_round",
        "Total Raised": "total_raised",
        "Date Added": "date_added",
    }
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    db_columns = list(column_mapping.values())
    df_final = df_renamed[[col for col in db_columns if col in df_renamed.columns]]
    
    # Convert DataFrame to list of dicts
    records = df_final.to_dict(orient="records")
    
    # Clean up NaN values
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, str) and value.strip() == "":
                record[key] = None
    
    # Insert data in batches
    batch_size = 50
    total_inserted = 0
    
    print(f"📤 Inserting {len(records)} records into Supabase...")
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            result = supabase.table("deals").insert(batch).execute()
            total_inserted += len(batch)
            print(f"  ✅ Inserted batch {i // batch_size + 1} ({len(batch)} records)")
        except Exception as e:
            print(f"  ❌ Error inserting batch {i // batch_size + 1}: {e}")
            # Try inserting one by one for this batch
            for j, record in enumerate(batch):
                try:
                    supabase.table("deals").insert(record).execute()
                    total_inserted += 1
                except Exception as e2:
                    print(f"    ❌ Failed to insert record {i + j + 1}: {e2}")
    
    print(f"\n🎉 Migration complete! {total_inserted}/{len(records)} records inserted.")


if __name__ == "__main__":
    migrate_csv_to_supabase()

