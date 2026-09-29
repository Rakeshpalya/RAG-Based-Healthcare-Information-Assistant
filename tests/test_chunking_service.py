import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.services.chunking_service import TextChunkingService


def test_empty_and_whitespace_text():
    """Test that empty or whitespace-only inputs return empty list."""
    assert TextChunkingService.chunk_text("") == []
    assert TextChunkingService.chunk_text("   \n\n\t  ") == []
    print("[PASS] test_empty_and_whitespace_text passed.")


def test_short_text():
    """Test text shorter than chunk size produces exactly 1 chunk."""
    short_text = "Patient John Doe presents with mild fever and headaches. Vitals are stable."
    chunks = TextChunkingService.chunk_text(short_text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chunk_0"
    assert chunks[0]["text"] == short_text
    assert chunks[0]["word_count"] == len(short_text.split())
    assert chunks[0]["character_count"] == len(short_text)
    assert chunks[0]["start_char_idx"] == 0
    print("[PASS] test_short_text passed.")


def test_long_text_multiple_chunks():
    """Test long medical text generates multiple chunks with specified size."""
    paragraphs = []
    for i in range(1, 15):
        paragraphs.append(
            f"Paragraph {i}: Clinical note section for patient case history {i}. "
            f"Symptom evaluation indicates mild hypertension and elevated blood glucose levels. "
            f"Recommended medication adjustment and follow-up lab work in 2 weeks."
        )
    long_doc = "\n\n".join(paragraphs)

    # Chunk with small chunk_size=40 words
    chunks = TextChunkingService.chunk_text(long_doc, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    for idx, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"chunk_{idx}"
        assert chunk["character_count"] > 0
        assert chunk["word_count"] > 0
        assert chunk["start_char_idx"] >= 0

    print(f"[PASS] test_long_text_multiple_chunks passed. Generated {len(chunks)} chunks.")


def test_chunk_overlap():
    """Test that consecutive chunks share overlapping context."""
    text_blocks = [f"Word_{i}" for i in range(1, 101)]
    text = " ".join(text_blocks)

    chunks = TextChunkingService.chunk_text(text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) >= 3

    # Check overlap between chunk 0 and chunk 1
    chunk_0_words = chunks[0]["text"].split()
    chunk_1_words = chunks[1]["text"].split()

    # The last words of chunk 0 should overlap with the beginning words of chunk 1
    overlap_count = len(set(chunk_0_words).intersection(set(chunk_1_words)))
    assert overlap_count >= 1, "Expected overlapping words between consecutive chunks"
    print(f"[PASS] test_chunk_overlap passed. Shared words: {overlap_count}")


def run_medical_document_chunking_demo():
    """Demonstration of chunking a multi-paragraph medical document."""
    medical_document = (
        "CLINICAL SUMMARY & DISCHARGE REPORT\n\n"
        "Patient Information: Jane Smith, 62-year-old female. Admitted for acute shortness of breath "
        "and bilateral lower extremity edema. Past medical history is significant for congestive heart failure (CHF), "
        "type 2 diabetes mellitus, and chronic kidney disease stage III.\n\n"
        "Hospital Course: Patient was placed on intravenous furosemide for diuresis, achieving a net negative fluid balance "
        "of 3.5 liters over 48 hours. Oxygen saturation improved from 88% on room air to 96% on 2L nasal cannula. "
        "Echocardiogram demonstrated an ejection fraction (EF) of 42% with mild mitral regurgitation.\n\n"
        "Laboratory Results: Initial NT-proBNP was elevated at 4,200 pg/mL, decreasing to 1,800 pg/mL at discharge. "
        "Serum creatinine stabilized at 1.4 mg/dL. Fasting blood glucose maintained between 110-140 mg/dL with sliding scale insulin.\n\n"
        "Discharge Instructions & Plan:\n"
        "1. Continue oral Furosemide 40mg daily.\n"
        "2. Resume Lisinopril 10mg daily and Metformin 500mg twice daily.\n"
        "3. Maintain strict 2-liter fluid restriction and low-sodium diet (<2,000 mg/day).\n"
        "4. Daily weight monitoring; call clinic if weight increases by >3 lbs in 24 hours.\n"
        "5. Follow-up appointment scheduled with Cardiology in 7 days."
    )

    print("\n" + "=" * 70)
    print("MEDICAL DOCUMENT TEXT CHUNKING DEMO")
    print("=" * 70)

    print(f"Original Document Character Count: {len(medical_document)}")
    print(f"Original Document Word Count: {len(medical_document.split())}\n")

    # Perform chunking with small chunk size for demonstration (50 words, 10 overlap)
    chunks = TextChunkingService.chunk_text(medical_document, chunk_size=50, chunk_overlap=10)

    for chunk in chunks:
        print(f"--- [{chunk['chunk_id'].upper()}] ---")
        print(f"Start Char Index : {chunk['start_char_idx']}")
        print(f"Character Count  : {chunk['character_count']}")
        print(f"Word Count       : {chunk['word_count']}")
        print(f"Text Content:\n{chunk['text']}\n")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_empty_and_whitespace_text()
    test_short_text()
    test_long_text_multiple_chunks()
    test_chunk_overlap()
    run_medical_document_chunking_demo()
    print("[SUCCESS] All TextChunkingService unit tests & demo passed successfully!")
