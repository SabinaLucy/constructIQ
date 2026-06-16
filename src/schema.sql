-- src/schema.sql — run once to initialise the database

CREATE TABLE projects (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    description TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE audit_jobs (
    id             SERIAL PRIMARY KEY,
    project_id     INTEGER REFERENCES projects(id),
    job_id         VARCHAR(64) UNIQUE NOT NULL,   -- matches DynamoDB key
    status         VARCHAR(32) DEFAULT 'pending', -- pending/processing/complete/failed
    photo_count    INTEGER,
    safety_score   FLOAT,
    risk_level     VARCHAR(16),                   -- LOW / ELEVATED / CRITICAL
    result_pdf_url TEXT,
    created_at     TIMESTAMP DEFAULT NOW(),
    completed_at   TIMESTAMP
);

CREATE TABLE safety_violations (
    id           SERIAL PRIMARY KEY,
    audit_job_id INTEGER REFERENCES audit_jobs(id),
    violation_type VARCHAR(128),
    severity       VARCHAR(32),
    photo_filename VARCHAR(255),
    bbox_json      TEXT,   -- bounding box coordinates as JSON string
    confidence     FLOAT,
    detected_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bulletins (
    id           SERIAL PRIMARY KEY,
    project_id   INTEGER REFERENCES projects(id),
    week_of      DATE NOT NULL,
    risk_level   VARCHAR(16),
    safety_score FLOAT,
    content      TEXT,   -- full GPT-4o bulletin text
    pdf_url      TEXT,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE rag_documents (
    id           SERIAL PRIMARY KEY,
    project_id   INTEGER REFERENCES projects(id),
    filename     VARCHAR(255),
    doc_type     VARCHAR(64),  -- contract / spec / inspection / schedule
    page_count   INTEGER,
    ingested_at  TIMESTAMP DEFAULT NOW(),
    chunk_count  INTEGER
);

-- Model results table (used by Model Dashboard, populated from Phase 2 onward)
CREATE TABLE model_results (
    id           SERIAL PRIMARY KEY,
    phase        VARCHAR(32),
    model_name   VARCHAR(128),
    accuracy     FLOAT,
    macro_f1     FLOAT,
    notes        TEXT,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- Indexes for the History page queries
CREATE INDEX idx_audit_jobs_project   ON audit_jobs(project_id, created_at DESC);
CREATE INDEX idx_bulletins_project    ON bulletins(project_id, week_of DESC);
CREATE INDEX idx_violations_audit     ON safety_violations(audit_job_id);
