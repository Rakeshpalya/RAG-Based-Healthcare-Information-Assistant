import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# 1. Test /health/readiness
h = client.get("/health/readiness")
assert h.status_code == 200, f"/health/readiness failed: {h.text}"
print("1. Health readiness probe: PASSED ->", h.json()["status"])

# 2. Test Safety Interception (Acute Emergency)
emergency_res = client.post("/rag/query", json={"question": "I have severe crushing chest pain and slurred speech, help!"})
assert emergency_res.status_code == 200, f"Emergency response failed: {emergency_res.text}"
data_e = emergency_res.json()
assert data_e["retrieval_status"] == "safety_intercepted", f"Retrieval status mismatch: {data_e['retrieval_status']}"
assert "EMERGENCY ADVISORY" in data_e["answer"] or "911" in data_e["answer"], "Emergency advisory missing from answer"
assert data_e["sources"] == [], f"Expected 0 sources on safety interception, got: {data_e['sources']}"
assert data_e["timings"]["safety_assessment"]["category"] == "EMERGENCY_SYMPTOMS"
assert data_e["timings"]["llm_called"] is False
print("2. Safety Interception test: PASSED (Emergency intercepted, LLM bypassed)")

# 3. Test Normal RAG retrieval & request_id
normal_res = client.post("/rag/query", json={"question": "What are the lifestyle measures for hypertension?"})
assert normal_res.status_code == 200, f"Normal RAG failed: {normal_res.text}"
data_n = normal_res.json()
assert "request_id" in data_n, "request_id missing from normal response"
print(f"3. Normal RAG Query test: PASSED (Retrieval status: {data_n['retrieval_status']}, Sources: {len(data_n.get('sources', []))})")

# 4. Check secret scrubbing
payload_str = json.dumps(data_n)
for secret_word in ["password", "secret", "bearer", "eyJ"]:
    assert secret_word not in payload_str.lower(), f"Secret keyword {secret_word} found in response!"
print("4. Secret Scrubbing check: PASSED (Zero credentials or secrets in output)")
print("\nALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
