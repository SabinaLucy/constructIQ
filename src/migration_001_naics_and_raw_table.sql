
ALTER TABLE safety_violations ADD COLUMN IF NOT EXISTS naics_code VARCHAR(10);

CREATE TABLE IF NOT EXISTS raw_osha_records (
    id          SERIAL PRIMARY KEY,
    source      VARCHAR(64),
    raw_json    TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_violations_naics ON safety_violations(naics_code);
CREATE INDEX IF NOT EXISTS idx_raw_osha_source ON raw_osha_records(source, ingested_at DESC);
