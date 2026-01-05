-- Supabase Table Schema for M&A Tracker
-- Run this in the Supabase SQL Editor to create your table

-- Create the deals table
CREATE TABLE IF NOT EXISTS deals (
    id SERIAL PRIMARY KEY,
    company_target VARCHAR(255),
    acquirer VARCHAR(255),
    category VARCHAR(255),
    amount VARCHAR(50),
    revenue VARCHAR(50),
    multiple VARCHAR(50),
    rationale TEXT,
    company_url VARCHAR(500),
    company_description TEXT,
    last_round_raised VARCHAR(50),
    valuation_of_last_round VARCHAR(50),
    total_raised VARCHAR(50),
    date_added TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on date_added for faster sorting
CREATE INDEX IF NOT EXISTS idx_deals_date_added ON deals(date_added DESC);

-- Create an index on company name for faster search
CREATE INDEX IF NOT EXISTS idx_deals_company_target ON deals(company_target);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE deals ENABLE ROW LEVEL SECURITY;

-- Create a policy that allows all authenticated users to read
CREATE POLICY "Allow authenticated read access" ON deals
    FOR SELECT
    TO authenticated
    USING (true);

-- Create a policy that allows all authenticated users to insert
CREATE POLICY "Allow authenticated insert" ON deals
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Create a policy that allows all authenticated users to update
CREATE POLICY "Allow authenticated update" ON deals
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Create a policy that allows all authenticated users to delete
CREATE POLICY "Allow authenticated delete" ON deals
    FOR DELETE
    TO authenticated
    USING (true);

-- Also allow anon access for service role operations (used by the app)
CREATE POLICY "Allow service role full access" ON deals
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

