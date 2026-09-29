# Phase 7: Frontend Architecture & Live RAG Chat Implementation Plan

**Project**: `AI-Healthcare-Agent`  
**Milestone**: Phase 7 — Frontend + Live RAG Chat + Observability UI  
**Target Architecture**: React 18 + TypeScript + Vite + Tailwind CSS + Lucide Icons  
**Primary Integration**: `POST /rag/query`, `GET /health/readiness`, `GET /health/liveness`  
**Strict Backend Invariant**: Zero client-side medical logic. The backend is the sole source of truth for safety, retrieval, citations, hallucination protection, confidence, and request IDs.

---

## 1. Existing API Audit & Integration Contracts

### 1.1 `POST /rag/query` (Primary Clinical RAG Chat Endpoint)
- **URL**: `${VITE_API_BASE_URL}/rag/query`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Payload (`RAGQueryRequest`)**:
  ```typescript
  export interface RAGQueryRequest {
    question: string;                 // Required, min length: 3, max length: 1000
    top_k?: number;                   // Optional, default: 5 (range: 1-20)
    similarity_threshold?: number;    // Optional, default: 0.25 (range: -1.0 to 1.0)
    conversation_history?: Array<{    // Optional contextual message chain
      role: 'user' | 'assistant';
      content: string;
    }>;
  }
  ```
- **Response Payload (`RAGQueryResponse`)**:
  ```typescript
  export interface SourceCitation {
    chunk_id: string;
    document_name?: string;
    document_id?: string;
    page_number?: number;
    similarity_score?: number;
    text: string;
  }

  export interface RAGQueryResponse {
    question: string;
    answer: string;
    retrieval_status: 'success' | 'no_relevant_context' | 'safety_intercepted' | 'service_unavailable';
    sources: SourceCitation[];
    context?: string | null;
    disclaimer: string;
    timings?: {
      total_time_ms?: number;
      embedding_time_ms?: number;
      search_time_ms?: number;
      generation_time_ms?: number;
      llm_called?: boolean;
      safety_category?: string;
      risk_level?: string;
      safety_assessment?: {
        category: string;
        risk_level: string;
        requires_escalation: boolean;
        allow_normal_rag: boolean;
      };
    };
    request_id?: string;
  }
  ```

### 1.2 `GET /health/readiness` & `GET /health/liveness` (System Status)
- **URL**: `${VITE_API_BASE_URL}/health/readiness`
- **Method**: `GET`
- **Response**:
  ```typescript
  export interface ReadinessResponse {
    status: 'ready' | 'not_ready';
    service: string;
    environment: string;
    checks: {
      vector_store: {
        status: 'ready' | 'degraded' | 'mismatched';
        vector_count: number;
        expected_count: number;
      };
      embedding_service: {
        status: 'ready' | 'degraded';
        dimension: number;
      };
      safety_engine: {
        status: 'ready' | 'degraded';
      };
    };
  }
  ```

### 1.3 Safety Interception Response Model
When an emergency or prohibited clinical condition is intercepted by `MedicalSafetyGuard`:
- `retrieval_status`: `"safety_intercepted"`
- `answer`: Contains prominent emergency directive (`"EMERGENCY ADVISORY: ... Call 911 immediately"`)
- `sources`: `[]` (Empty)
- `timings.safety_assessment.category`: `"EMERGENCY_SYMPTOMS"` (or related critical category)
- `timings.llm_called`: `false`
- **Frontend Action**: Render prominent emergency banner (`bg-rose-50`, `border-rose-300`, `text-rose-900`) advising immediate professional emergency care.

---

## 2. Frontend Architecture & Technology Stack

```
AI-Healthcare-Agent/
├── src/
│   ├── api/
│   │   ├── client.ts             # Axios/fetch HTTP client with interceptors & timeout
│   │   └── ragApi.ts             # Typed functions: queryRAG(), checkReadiness()
│   ├── components/
│   │   ├── ChatArea.tsx          # Main scrollable message stream
│   │   ├── ChatInput.tsx         # Accessible textarea (Enter=Send, Shift+Enter=Newline)
│   │   ├── ChatMessage.tsx       # Message bubble (User vs Assistant)
│   │   ├── CitationCard.tsx      # Expandable source citation card with excerpt
│   │   ├── EmergencyBanner.tsx   # Prominent clinical emergency warning
│   │   ├── Header.tsx            # App branding, readiness indicator, clear chat
│   │   ├── MedicalDisclaimer.tsx # Persistent, unobtrusive medical notice
│   │   ├── RequestIdBadge.tsx    # Subtle request ID badge with one-click copy
│   │   └── Sidebar.tsx           # Conversation history list, new chat, delete chat
│   ├── types/
│   │   └── index.ts              # TypeScript interfaces for API and state
│   ├── utils/
│   │   ├── storage.ts            # LocalStorage conversation session persistence
│   │   └── formatters.ts         # Timestamp, score, and citation text formatting
│   ├── App.tsx                   # Root state orchestration (sessions, active chat)
│   ├── main.tsx                  # React 18 DOM mount point
│   └── index.css                 # Tailwind CSS directives & healthcare theme tokens
├── public/                       # Static public assets
├── tests/
│   └── frontend/
│       ├── chat.test.tsx         # Comprehensive Vitest component tests
│       └── ragApi.test.ts        # Unit tests for API client
├── package.json                  # Dependencies (React 18, Vite, Tailwind, Lucide)
├── tsconfig.json                 # Strict TypeScript configuration
├── tailwind.config.js            # Curated clinical color palette (slate, sky, emerald, rose)
└── vite.config.ts                # Vite build configuration with test setup
```

---

## 3. UI/UX Design System for Healthcare

- **Theme Palette**:
  - Background: `bg-slate-50` / `bg-white` (High contrast, clean clinical background)
  - Primary / Brand: `sky-600` / `blue-700` (Professional, trustworthy medical blue)
  - Success / Ready: `emerald-500` / `emerald-700` (Readiness badge, grounded citations)
  - Emergency / Critical: `rose-600` / `rose-700` (High-priority emergency alerts)
  - Text: `text-slate-900` (WCAG AAA compliant contrast ratio > 7:1)
- **Accessibility & Semantics**:
  - Full keyboard navigability: `Tab`, `Shift+Tab`, `Enter` to send, `Escape` to close modals.
  - Visible focus indicators: `focus:ring-2 focus:ring-sky-500 focus:outline-none`.
  - ARIA attributes: `role="log"`, `aria-live="polite"` for incoming assistant messages.
  - Screen-reader friendly announcements for readiness and loading indicators.

---

## 4. Key Functional Features

1. **Dual-Pane Layout**: Collapsible conversation history sidebar + full-width clinical chat panel.
2. **Local Session Management**: Multi-chat support stored securely in `localStorage` without leaking sensitive patient data to third parties.
3. **Structured Source Cards**: Expandable clinical citation badges linking `[Source 1]` directly to excerpt chunks with document name and relevance score.
4. **Traceable Observability**: Every assistant message renders a subtle `req_...` badge with a 1-click clipboard copy utility for audit and debugging.
5. **Zero-Latency Emergency Interception**: Real-time styling transformation if the backend detects emergency symptoms or clinical contraindications.
6. **Robust Error Resilience**: Automatic detection of offline backend, network timeouts (15s timeout threshold), and graceful friendly recovery prompts without technical jargon or stack traces.

---

## 5. Implementation Roadmap

- [x] Step 1: Complete API audit and create `PHASE_7_FRONTEND_PLAN.md`.
- [ ] Step 2: Configure `package.json`, `tsconfig.json`, `vite.config.ts`, and `tailwind.config.js`.
- [ ] Step 3: Implement TypeScript types and `src/api/ragApi.ts`.
- [ ] Step 4: Implement UI components (`Header`, `Sidebar`, `ChatArea`, `CitationCard`, `EmergencyBanner`, `RequestIdBadge`).
- [ ] Step 5: Implement `App.tsx` state management and auto-scroll behavior.
- [ ] Step 6: Enable FastAPI CORS middleware in `backend/main.py`.
- [ ] Step 7: Write comprehensive frontend tests (`tests/frontend/`).
- [ ] Step 8: Build production frontend bundle (`npm run build`).
- [ ] Step 9: Run complete backend regression (`508+ tests`) and evaluation runner.
- [ ] Step 10: Verify vector store invariant (744 FAISS vectors == 744 metadata records, SHA256 match).
- [ ] Step 11: Document deliverables in `PHASE_7_FRONTEND_REPORT.md` and update `README.md`.
