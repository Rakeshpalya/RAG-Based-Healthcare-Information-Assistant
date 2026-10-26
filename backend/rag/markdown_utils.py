"""
Markdown utilities and sanitization for AI Healthcare Agent.
Ensures responses are free of SVG artifacts, localhost anchor URLs, and malformed tags
while preserving genuine clinical markdown (headings, bullets, bold text, citations).
"""

import re
from typing import Optional


def clean_ai_markdown(text: Optional[str]) -> str:
    """
    Cleans AI-generated Markdown and Streamlit-rendered responses to remove
    unwanted SVG anchor links, localhost URLs, internal application URLs, raw HTML/SVG artifacts,
    broken Markdown links, and malformed tags while strictly preserving genuine clinical content:
      - Headings (### What Is Hypertension?)
      - Bullet lists (- item)
      - Numbered lists (1. item)
      - Bold / Italics (**text**, *text*)
      - Citations ([Source 1])
      - Paragraphs, medical disclaimers, and emergency advisories
    """
    if not text:
        return ""

    cleaned = str(text)

    # 1. Remove markdown reference link definitions for SVG or anchor links
    cleaned = re.sub(r'^\s*!?\[\s*svg\s*\]\s*:\s*.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'^\s*\[[^\]]*\]\s*:\s*https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/[^\s]*)?$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'^\s*\[[^\]]*\]\s*:\s*#[^\s]*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # 2. Remove markdown links whose link text is 'svg' (using a space to prevent merging)
    cleaned = re.sub(r'!?\[\s*svg\s*\]\s*\([^\)]*\)', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'!?\[\s*svg\s*\]\s*(?:\[[^\]]*\])?', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'!?\[\s*svg\s*\]', ' ', cleaned, flags=re.IGNORECASE)

    # 3. Remove localhost or anchor markdown links preserving anchor text
    cleaned = re.sub(r'\[([^\]]+)\]\s*\(\s*https?://(?:localhost|127\.0\.0\.1)[^\)]*\)', r' \1 ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[([^\]]+)\]\s*\(\s*#[^\)]*\)', r' \1 ', cleaned, flags=re.IGNORECASE)

    # 4. Remove empty or broken markdown links: [](), [text]() -> text
    cleaned = re.sub(r'\[\s*\]\s*\(\s*\)', ' ', cleaned)
    cleaned = re.sub(r'\[([^\]]+)\]\s*\(\s*\)', r' \1 ', cleaned)

    # 5. Remove standalone localhost / internal application URLs
    cleaned = re.sub(r'https?://(?:localhost|127\.0\.0\.1)(?::\d+)?[^\s\)]*', ' ', cleaned, flags=re.IGNORECASE)

    # 6. Remove raw HTML anchor links and SVG tags (with space boundary)
    cleaned = re.sub(r'<a\s+[^>]*class=[\'"][^\'"]*(?:anchor|heading)[^\'"]*[\'"][^>]*>.*?</a>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<a\s+[^>]*aria-label=[\'"][^\'"]*heading[^\'"]*[\'"][^>]*>.*?</a>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<a\s+[^>]*href=[\'"][^\'"]*#[^\'"]*[\'"][^>]*>\s*<svg[^>]*>.*?</svg>\s*</a>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<a\s+[^>]*href=[\'"]https?://(?:localhost|127\.0\.0\.1)[^\'"]*[\'"][^>]*>.*?</a>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<svg[^>]*>.*?</svg>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<svg[^>]*/>', ' ', cleaned, flags=re.IGNORECASE)

    # 7. Strip arbitrary internal HTML wrappers (span, div, p, a, etc.) preserving inner text with word boundaries
    cleaned = re.sub(r'</?(?:div|span|section|header|footer|p|a|em|strong|b|i)[^>]*>', ' ', cleaned, flags=re.IGNORECASE)

    # 8. Remove standalone 'svg' lines or trailing 'svg' artifacts
    cleaned = re.sub(r'^\s*\[?\s*svg\s*\]?\s*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'(?:\s*\[?\s*svg\s*\]?:?)+$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bsvg\b', '', cleaned, flags=re.IGNORECASE)

    # 9. Clean up heading lines with leading/trailing artifact whitespace
    cleaned = re.sub(r'^(#{1,6})\s+', r'\1 ', cleaned, flags=re.MULTILINE)

    # 10. Ensure clean spacing around citation brackets and normalize 'Source1' -> 'Source 1'
    cleaned = re.sub(r'\[Source(\d+)\]', r'[Source \1]', cleaned)
    cleaned = re.sub(r'\bSource(\d+)\b', r'Source \1', cleaned)
    cleaned = re.sub(r'([A-Za-z0-9])(\[(?:Source\s+\d+|[Ss]ources?\b))', r'\1 \2', cleaned)
    cleaned = re.sub(r'(\[(?:Source\s+\d+|[Ss]ources?\b[^\]]*)\])([.,;:!?])?([A-Za-z0-9])', r'\1\2 \3', cleaned)

    # 11. Normalize punctuation spacing where words or sentences were accidentally conjoined
    cleaned = re.sub(r'([a-z0-9\]\)])([.!?])([A-Z])', r'\1\2 \3', cleaned)
    cleaned = re.sub(r'([a-z0-9\]\)])([,;:])([A-Za-z])', r'\1\2 \3', cleaned)
    cleaned = re.sub(r'\bExcessbody\b', 'Excess body', cleaned)
    cleaned = re.sub(r'\bExcessweight\b', 'Excess weight', cleaned)

    # 12. Ensure list and bullet points start on newlines to prevent broken inline lists (preserving bold **)
    cleaned = re.sub(r'([.!?:\n])\s*([*\-•])(?!\*)\s*', r'\1\n\2 ', cleaned)
    cleaned = re.sub(r'([.!?:\n])\s*(\d+[\.\)])\s*', r'\1\n\2 ', cleaned)
    cleaned = re.sub(r'([^\n\s])\s*([•])\s*', r'\1\n\2 ', cleaned)
    cleaned = re.sub(r'([^\n\s])\s+([*\-])(?!\*)\s*([A-Za-z0-9])', r'\1\n\2 \3', cleaned)
    cleaned = re.sub(r'([a-z0-9])\s*([*\-])(?!\*)\s*([A-Za-z0-9]+)', r'\1\n\2 \3', cleaned)

    # 13. Remove empty orphan bullet points and leftover artifact markers
    cleaned = re.sub(r'^\s*[-*•](?:\s+[-*•])*\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*[-*•\s]+$', '', cleaned, flags=re.MULTILINE)

    # 14. Normalize horizontal whitespace without destroying paragraphs
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r' +(?=\n)', '', cleaned)
    cleaned = re.sub(r'(?<=\n) +', '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()

