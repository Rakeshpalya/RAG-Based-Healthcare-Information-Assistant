# Healthcare AI Assistant — Streamlit Web Frontend (Phase 11)

This directory contains the Streamlit web frontend for the **AI Healthcare Research & Patient Assistance Agent**.

---

## 🎯 Architecture Overview

The frontend is a dedicated presentation layer that communicates with the existing FastAPI backend via standard REST HTTP endpoints. **No business logic, RAG retrieval, vector search, or direct LLM calls reside in the frontend.**

```
Streamlit Web UI (Port 8501)
         ↓ HTTP REST
FastAPI Backend (Port 8000)
         ↓
Orchestrator & Safety Agents
         ↓
RAG Retrieval & FAISS Vector Index (384-dim)
         ↓
Google Gemini 2.5 Flash
         ↓
PostgreSQL / Relational Metadata
```

---

## 📁 Directory Structure

```
frontend/
├── app.py                     # Streamlit application entrypoint & routing
├── api_client.py              # Centralized HTTP API client with timeouts & error handling
├── components/
│   ├── __init__.py
│   ├── sidebar.py             # Navigation, branding & live backend health status
│   ├── status.py              # Status badges (PROCESSED, UPLOADED, FAILED, CONNECTED)
│   ├── document_card.py       # Clean cards displaying document metadata & chunk counts
│   ├── source_card.py         # Grounded citations & retrieval relevance cards
│   └── chat.py                # Chat rendering & message history component
├── pages/
│   ├── __init__.py
│   ├── dashboard.py           # Metrics, recent docs/conversations, and quick actions
│   ├── documents.py           # PDF file uploader, status tracking & document viewer
│   └── history.py             # Conversation browser with chronological message threads
└── utils/
    ├── __init__.py
    └── helpers.py             # Date formatting, token estimation, text truncations
```

---

## 🚀 How to Run

### 1. Start the FastAPI Backend (Terminal 1)
```bash
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Streamlit Frontend (Terminal 2)
```bash
.\venv\Scripts\streamlit.exe run frontend/app.py
```
By default, the Streamlit interface will open in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `BACKEND_URL` | `http://127.0.0.1:8000` | URL of the running FastAPI backend server. |

---

## 🧪 Testing

Frontend unit tests use mocked backend API responses and can run without a live Gemini key or database server:
```bash
.\venv\Scripts\python.exe tests/test_frontend_client.py
```
Or using pytest:
```bash
.\venv\Scripts\python.exe -m pytest tests/test_frontend_client.py
```
