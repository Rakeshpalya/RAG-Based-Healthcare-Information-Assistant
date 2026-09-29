-- ==============================================================================
-- Supabase PostgreSQL Row Level Security (RLS) Policies
-- Project: AI-Healthcare-Agent
-- Phase: 12, Step 8 — Defense-in-Depth Database Security
-- ==============================================================================
--
-- ARCHITECTURAL NOTICE:
-- 1. FastAPI Application Layer:
--    The application connects to Supabase PostgreSQL using SQLAlchemy with the
--    'postgres' role (database owner/superuser). In PostgreSQL, table owners bypass
--    RLS by default. FastAPI's cryptographic JWT verification and repository
--    authorization filters remain the authoritative security boundary for backend queries.
--
-- 2. Database Defense-in-Depth:
--    Enabling RLS on these public tables protects against direct, unauthorized access
--    via Supabase's public PostgREST API (/rest/v1/), Supabase JavaScript/Python Client SDKs,
--    or any anonymous/authenticated client roles (anon, authenticated).
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Enable Row Level Security on all application tables
-- ------------------------------------------------------------------------------

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

-- ------------------------------------------------------------------------------
-- 2. USERS Table Policies
-- ------------------------------------------------------------------------------
-- Ensure authenticated users can only view or modify their own profile row.
-- Maps Supabase auth.jwt() email claim to application users.email.

DROP POLICY IF EXISTS "users_select_own" ON public.users;
CREATE POLICY "users_select_own" ON public.users
    FOR SELECT
    TO authenticated
    USING (
        email = auth.jwt() ->> 'email'
    );

DROP POLICY IF EXISTS "users_update_own" ON public.users;
CREATE POLICY "users_update_own" ON public.users
    FOR UPDATE
    TO authenticated
    USING (
        email = auth.jwt() ->> 'email'
    );

-- ------------------------------------------------------------------------------
-- 3. DOCUMENTS Table Policies
-- ------------------------------------------------------------------------------
-- Users can only select, insert, update, or delete documents they own.

DROP POLICY IF EXISTS "documents_select_own" ON public.documents;
CREATE POLICY "documents_select_own" ON public.documents
    FOR SELECT
    TO authenticated
    USING (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "documents_insert_own" ON public.documents;
CREATE POLICY "documents_insert_own" ON public.documents
    FOR INSERT
    TO authenticated
    WITH CHECK (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "documents_update_own" ON public.documents;
CREATE POLICY "documents_update_own" ON public.documents
    FOR UPDATE
    TO authenticated
    USING (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "documents_delete_own" ON public.documents;
CREATE POLICY "documents_delete_own" ON public.documents
    FOR DELETE
    TO authenticated
    USING (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

-- ------------------------------------------------------------------------------
-- 4. DOCUMENT_CHUNKS Table Policies
-- ------------------------------------------------------------------------------
-- Chunks inherit ownership directly from their parent document.

DROP POLICY IF EXISTS "chunks_select_parent_owner" ON public.document_chunks;
CREATE POLICY "chunks_select_parent_owner" ON public.document_chunks
    FOR SELECT
    TO authenticated
    USING (
        document_id IN (
            SELECT d.id FROM public.documents d
            JOIN public.users u ON d.user_id = u.id
            WHERE u.email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "chunks_insert_parent_owner" ON public.document_chunks;
CREATE POLICY "chunks_insert_parent_owner" ON public.document_chunks
    FOR INSERT
    TO authenticated
    WITH CHECK (
        document_id IN (
            SELECT d.id FROM public.documents d
            JOIN public.users u ON d.user_id = u.id
            WHERE u.email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "chunks_delete_parent_owner" ON public.document_chunks;
CREATE POLICY "chunks_delete_parent_owner" ON public.document_chunks
    FOR DELETE
    TO authenticated
    USING (
        document_id IN (
            SELECT d.id FROM public.documents d
            JOIN public.users u ON d.user_id = u.id
            WHERE u.email = auth.jwt() ->> 'email'
        )
    );

-- ------------------------------------------------------------------------------
-- 5. CONVERSATIONS Table Policies
-- ------------------------------------------------------------------------------
-- Users can only select, insert, update, or delete conversations they own.

DROP POLICY IF EXISTS "conversations_select_own" ON public.conversations;
CREATE POLICY "conversations_select_own" ON public.conversations
    FOR SELECT
    TO authenticated
    USING (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "conversations_insert_own" ON public.conversations;
CREATE POLICY "conversations_insert_own" ON public.conversations
    FOR INSERT
    TO authenticated
    WITH CHECK (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "conversations_update_own" ON public.conversations;
CREATE POLICY "conversations_update_own" ON public.conversations
    FOR UPDATE
    TO authenticated
    USING (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "conversations_delete_own" ON public.conversations;
CREATE POLICY "conversations_delete_own" ON public.conversations
    FOR DELETE
    TO authenticated
    USING (
        user_id IN (
            SELECT id FROM public.users WHERE email = auth.jwt() ->> 'email'
        )
    );

-- ------------------------------------------------------------------------------
-- 6. MESSAGES Table Policies
-- ------------------------------------------------------------------------------
-- Messages inherit ownership directly from their parent conversation.

DROP POLICY IF EXISTS "messages_select_parent_owner" ON public.messages;
CREATE POLICY "messages_select_parent_owner" ON public.messages
    FOR SELECT
    TO authenticated
    USING (
        conversation_id IN (
            SELECT c.id FROM public.conversations c
            JOIN public.users u ON c.user_id = u.id
            WHERE u.email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "messages_insert_parent_owner" ON public.messages;
CREATE POLICY "messages_insert_parent_owner" ON public.messages
    FOR INSERT
    TO authenticated
    WITH CHECK (
        conversation_id IN (
            SELECT c.id FROM public.conversations c
            JOIN public.users u ON c.user_id = u.id
            WHERE u.email = auth.jwt() ->> 'email'
        )
    );

DROP POLICY IF EXISTS "messages_delete_parent_owner" ON public.messages;
CREATE POLICY "messages_delete_parent_owner" ON public.messages
    FOR DELETE
    TO authenticated
    USING (
        conversation_id IN (
            SELECT c.id FROM public.conversations c
            JOIN public.users u ON c.user_id = u.id
            WHERE u.email = auth.jwt() ->> 'email'
        )
    );
