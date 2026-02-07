-- =============================================================================
-- ShelfSense AI - Database Initialization
-- =============================================================================
-- Run on first deployment to set up the database schema and extensions
-- =============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity search

-- Create custom types
DO $$ BEGIN
    CREATE TYPE read_status AS ENUM ('unread', 'reading', 'read', 'abandoned');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- =============================================================================
-- Books Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Core metadata
    title VARCHAR(500) NOT NULL,
    authors TEXT[] DEFAULT '{}',
    isbn_10 VARCHAR(10),
    isbn_13 VARCHAR(13),
    
    -- Extended metadata
    publisher VARCHAR(255),
    publish_date DATE,
    page_count INTEGER,
    language VARCHAR(10) DEFAULT 'en',
    description TEXT,
    
    -- Classification
    genres TEXT[] DEFAULT '{}',
    subjects TEXT[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    
    -- Images
    cover_url VARCHAR(500),
    thumbnail_url VARCHAR(500),
    spine_image_path VARCHAR(500),
    
    -- User data
    read_status read_status DEFAULT 'unread',
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    notes TEXT,
    date_added TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    date_read TIMESTAMP WITH TIME ZONE,
    
    -- Detection metadata
    detection_confidence FLOAT,
    ocr_text TEXT,
    
    -- System fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    
    -- External IDs
    google_books_id VARCHAR(50),
    open_library_id VARCHAR(50),
    
    -- Constraints
    CONSTRAINT valid_isbn_10 CHECK (isbn_10 IS NULL OR LENGTH(isbn_10) = 10),
    CONSTRAINT valid_isbn_13 CHECK (isbn_13 IS NULL OR LENGTH(isbn_13) = 13)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_books_title ON books USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_books_authors ON books USING gin (authors);
CREATE INDEX IF NOT EXISTS idx_books_isbn_10 ON books (isbn_10) WHERE isbn_10 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_books_isbn_13 ON books (isbn_13) WHERE isbn_13 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_books_genres ON books USING gin (genres);
CREATE INDEX IF NOT EXISTS idx_books_read_status ON books (read_status);
CREATE INDEX IF NOT EXISTS idx_books_rating ON books (rating) WHERE rating IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_books_date_added ON books (date_added DESC);
CREATE INDEX IF NOT EXISTS idx_books_deleted ON books (deleted_at) WHERE deleted_at IS NULL;

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_books_fts ON books USING gin (
    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(array_to_string(authors, ' '), '') || ' ' || coalesce(description, ''))
);

-- =============================================================================
-- Embeddings Table (for hybrid search alongside FAISS)
-- =============================================================================
CREATE TABLE IF NOT EXISTS book_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    
    embedding_type VARCHAR(50) NOT NULL,  -- 'text', 'visual', 'fused'
    model_name VARCHAR(100) NOT NULL,
    vector_dim INTEGER NOT NULL,
    
    -- Store as binary for efficiency (use pgvector extension in production)
    embedding BYTEA NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(book_id, embedding_type)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_book ON book_embeddings (book_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON book_embeddings (embedding_type);

-- =============================================================================
-- Conversations Table (RAG chat history)
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    title VARCHAR(255),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    total_messages INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at DESC);

-- =============================================================================
-- Messages Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    
    -- RAG metadata
    retrieved_book_ids UUID[] DEFAULT '{}',
    citations JSONB DEFAULT '[]',
    
    -- Usage tracking
    token_count INTEGER,
    latency_ms INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, created_at);

-- =============================================================================
-- Detection Jobs Table (async processing)
-- =============================================================================
CREATE TABLE IF NOT EXISTS detection_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    status VARCHAR(20) NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    
    -- Input
    image_count INTEGER NOT NULL DEFAULT 1,
    
    -- Results
    books_detected INTEGER DEFAULT 0,
    books_identified INTEGER DEFAULT 0,
    books_added INTEGER DEFAULT 0,
    results JSONB,
    error TEXT,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Computed duration
    duration_ms INTEGER GENERATED ALWAYS AS (
        EXTRACT(MILLISECONDS FROM (completed_at - started_at))::INTEGER
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON detection_jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON detection_jobs (created_at DESC);

-- =============================================================================
-- Reading Sessions Table (analytics)
-- =============================================================================
CREATE TABLE IF NOT EXISTS reading_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    
    pages_read INTEGER,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_book ON reading_sessions (book_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON reading_sessions (started_at DESC);

-- =============================================================================
-- Triggers
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER books_updated_at
    BEFORE UPDATE ON books
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Update conversation message count
CREATE OR REPLACE FUNCTION update_conversation_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversations 
    SET 
        total_messages = (SELECT COUNT(*) FROM messages WHERE conversation_id = NEW.conversation_id),
        total_tokens = (SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = NEW.conversation_id)
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER messages_stats_update
    AFTER INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_stats();

-- =============================================================================
-- Initial Data (optional)
-- =============================================================================

-- You can add initial data here if needed
-- INSERT INTO books (title, authors, isbn_13) VALUES ...

-- =============================================================================
-- Permissions
-- =============================================================================

-- Grant permissions to application user (adjust username as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO shelfsense;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO shelfsense;
