import re
from typing import List, Dict, Any


class TextChunkingService:
    """
    Service for splitting medical documents into semantically coherent text chunks
    with boundary preservation and sliding window overlap.
    """

    DEFAULT_CHUNK_SIZE = 500       # Target chunk size in approximate words/tokens
    DEFAULT_CHUNK_OVERLAP = 50     # Target overlap size in approximate words/tokens

    @classmethod
    def _split_into_units(cls, text: str, max_words: int) -> List[str]:
        """
        Recursively breaks text into paragraphs, sentences, and words to ensure
        no single atomic unit exceeds max_words.
        """
        # Step 1: Split into paragraphs by double newlines or single newlines
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        units: List[str] = []

        for paragraph in paragraphs:
            para_words = paragraph.split()
            if len(para_words) <= max_words:
                units.append(paragraph)
            else:
                # Step 2: Split oversized paragraph into sentences
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if len(sentence_words) <= max_words:
                        units.append(sentence)
                    else:
                        # Step 3: Split oversized sentence into word sub-blocks
                        words = sentence_words
                        for i in range(0, len(words), max_words):
                            sub_block = " ".join(words[i:i + max_words])
                            if sub_block.strip():
                                units.append(sub_block.strip())
        return units

    @classmethod
    def chunk_text(
        cls,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks preserving paragraph and sentence boundaries.

        Args:
            text: Cleaned text string to chunk.
            chunk_size: Target maximum word count per chunk (default 500).
            chunk_overlap: Target overlapping word count between consecutive chunks (default 50).

        Returns:
            List of dictionaries containing chunk_id, text, character_count, word_count, and start_char_idx.
        """
        if not text or not text.strip():
            return []

        clean_input = text.strip()

        # Sanitize parameters
        chunk_size = max(10, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        # Break document into atomic sentence/paragraph units
        units = cls._split_into_units(clean_input, chunk_size)
        if not units:
            return []

        chunks: List[Dict[str, Any]] = []
        current_units: List[str] = []
        current_word_count = 0
        search_offset = 0

        for unit in units:
            unit_words = len(unit.split())

            # Check if adding unit exceeds chunk_size
            if current_units and (current_word_count + unit_words > chunk_size):
                # Build current chunk text
                chunk_text = "\n\n".join(current_units) if "\n" not in current_units[0] else " ".join(current_units)
                chunk_text = chunk_text.strip()

                if chunk_text:
                    start_idx = clean_input.find(chunk_text[:min(50, len(chunk_text))], search_offset)
                    if start_idx == -1:
                        start_idx = search_offset

                    chunk_id = f"chunk_{len(chunks)}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "character_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "start_char_idx": start_idx
                    })
                    search_offset = max(search_offset, start_idx + 1)

                # Compute overlap for the next chunk
                overlap_units: List[str] = []
                overlap_words = 0
                for prev_unit in reversed(current_units):
                    p_words = len(prev_unit.split())
                    if overlap_words + p_words <= chunk_overlap or not overlap_units:
                        overlap_units.insert(0, prev_unit)
                        overlap_words += p_words
                    else:
                        break

                current_units = overlap_units + [unit]
                current_word_count = sum(len(u.split()) for u in current_units)
            else:
                current_units.append(unit)
                current_word_count += unit_words

        # Handle remaining units
        if current_units:
            chunk_text = " ".join(current_units).strip()
            if chunk_text:
                start_idx = clean_input.find(chunk_text[:min(50, len(chunk_text))], search_offset)
                if start_idx == -1:
                    start_idx = search_offset

                chunk_id = f"chunk_{len(chunks)}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "start_char_idx": start_idx
                })

        return chunks
