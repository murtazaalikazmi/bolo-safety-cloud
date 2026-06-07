-- ─────────────────────────────────────────────────────────────
-- Bolo Safety Cloud — Supabase Database Setup
-- Run this once in Supabase → SQL Editor
-- ─────────────────────────────────────────────────────────────

-- Observations table (stores all processed HSE reports)
CREATE TABLE IF NOT EXISTS observations (
    id                   BIGSERIAL PRIMARY KEY,
    time_of_reporting    TIMESTAMPTZ,
    name                 TEXT,
    urdu_observation     TEXT,
    english_translation  TEXT,
    hse_categorization   TEXT,
    severity             TEXT,
    location             TEXT,
    ai_reasoning         TEXT,
    audio_file           TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Processed files log (prevents reprocessing the same audio)
CREATE TABLE IF NOT EXISTS processed_files (
    filename     TEXT PRIMARY KEY,
    skipped      BOOLEAN DEFAULT FALSE,
    reason       TEXT DEFAULT '',
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (recommended for Supabase)
ALTER TABLE observations    ENABLE ROW LEVEL SECURITY;
ALTER TABLE processed_files ENABLE ROW LEVEL SECURITY;

-- Allow the anon key full access (the app handles auth itself via passwords)
CREATE POLICY "allow_all_observations"    ON observations    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_processed_files" ON processed_files FOR ALL USING (true) WITH CHECK (true);
