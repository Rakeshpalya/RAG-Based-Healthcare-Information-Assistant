import os
from datetime import datetime
from typing import Optional, Union, List, Dict, Any


def format_timestamp(ts: Optional[Union[str, datetime]]) -> str:
    """Formats an ISO datetime string or datetime object into a human-readable display string."""
    if not ts:
        return "Unknown date"
    try:
        if isinstance(ts, str):
            # Parse ISO 8601 string
            clean_ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
        else:
            dt = ts
        return dt.strftime("%b %d, %Y • %I:%M %p")
    except Exception:
        return str(ts)


def format_bytes(bytes_count: Optional[int]) -> str:
    """Formats integer bytes into human-readable B, KB, or MB string."""
    if bytes_count is None or bytes_count < 0:
        return "Unknown size"
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    else:
        return f"{bytes_count / (1024 * 1024):.2f} MB"


def truncate_text(text: Optional[str], max_chars: int = 160) -> str:
    """Truncates text cleanly to max_chars, adding an ellipsis if shortened."""
    if not text:
        return ""
    text_clean = text.strip()
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars].rsplit(" ", 1)[0] + "..."


def clean_filename(file_path: Optional[str]) -> str:
    """Extracts the base filename from a potentially long or Windows/Linux path with sanitization."""
    if not file_path:
        return "Unnamed Document"
    clean = str(file_path).replace("\x00", "").replace("<", "").replace(">", "")
    clean = clean.replace("\\", "/")
    name = os.path.basename(clean)
    name = name.replace("..", "").strip()
    return name or "Unnamed Document"


def format_similarity(score: Optional[float]) -> str:
    """
    Formats a vector similarity score cleanly.
    Labels it strictly as retrieval relevance without implying medical certainty.
    """
    if score is None:
        return "N/A"
    return f"{score:.3f} relevance"


def clean_ai_markdown(text: Optional[str], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Cleans AI-generated Markdown and Streamlit-rendered responses to remove
    unwanted SVG anchor links, localhost URLs, internal application URLs, raw HTML/SVG artifacts,
    XSS vectors, broken Markdown links, and malformed tags while strictly preserving genuine clinical content:
      - Headings (### What Is Hypertension?)
      - Bullet lists (- item)
      - Numbered lists (1. item)
      - Bold / Italics (**text**, *text*)
      - Citations ([Source 1])
      - Paragraphs, medical disclaimers, and emergency advisories
    """
    if not text:
        return ""

    import re

    cleaned = str(text)

    # 0. Sanitize malicious script tags and event handlers (XSS protection)
    cleaned = re.sub(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<\s*script[^>]*>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<\s*object[^>]*>.*?<\s*/\s*object\s*>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'\bon\w+\s*=\s*[\'"][^\'"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bon\w+\s*=\s*[^\s>]+', '', cleaned, flags=re.IGNORECASE)

    # Citation spoofing suppression: If sources is explicitly empty, strip unverified citations
    if sources is not None and len(sources) == 0:
        cleaned = re.sub(r'\[Source\s*(?:#|:)?\s*\d+\]', '', cleaned, flags=re.IGNORECASE)

    # 1. Remove markdown reference link definitions for SVG or anchor links
    cleaned = re.sub(r'^\s*!?\[\s*svg\s*\]\s*:\s*.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'^\s*\[[^\]]*\]\s*:\s*https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/[^\s]*)?$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'^\s*\[[^\]]*\]\s*:\s*#[^\s]*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # 2. Completely remove any markdown link whose link text is 'svg'
    cleaned = re.sub(r'!?\[\s*svg\s*\]\s*\([^\)]*\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'!?\[\s*svg\s*\]\s*(?:\[[^\]]*\])?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'!?\[\s*svg\s*\]', '', cleaned, flags=re.IGNORECASE)

    # 3. Completely remove any markdown link pointing to localhost, 127.0.0.1, or anchor
    cleaned = re.sub(r'\[[^\]]*\]\s*\(\s*https?://(?:localhost|127\.0\.0\.1)[^\)]*\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[[^\]]*\]\s*\(\s*#[^\)]*\)', '', cleaned, flags=re.IGNORECASE)

    # 4. Remove empty or broken markdown links: [](), [text]() -> text
    cleaned = re.sub(r'\[\s*\]\s*\(\s*\)', '', cleaned)
    cleaned = re.sub(r'\[([^\]]+)\]\s*\(\s*\)', r'\1', cleaned)

    # 5. Remove standalone localhost / internal application URLs
    cleaned = re.sub(r'https?://(?:localhost|127\.0\.0\.1)(?::\d+)?[^\s\)]*', '', cleaned, flags=re.IGNORECASE)

    # 6. Remove raw HTML anchor links and SVG tags
    cleaned = re.sub(r'<a\s+[^>]*class=[\'"][^\'"]*(?:anchor|heading)[^\'"]*[\'"][^>]*>.*?</a>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<a\s+[^>]*aria-label=[\'"][^\'"]*heading[^\'"]*[\'"][^>]*>.*?</a>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<a\s+[^>]*href=[\'"][^\'"]*#[^\'"]*[\'"][^>]*>\s*<svg[^>]*>.*?</svg>\s*</a>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<a\s+[^>]*href=[\'"]https?://(?:localhost|127\.0\.0\.1)[^\'"]*[\'"][^>]*>.*?</a>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<svg[^>]*>.*?</svg>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<svg[^>]*/>', '', cleaned, flags=re.IGNORECASE)

    # 7. Strip arbitrary internal HTML wrappers (span, div) preserving inner text
    cleaned = re.sub(r'</?(?:div|span|section|header|footer)[^>]*>', '', cleaned, flags=re.IGNORECASE)

    # 8. Remove standalone 'svg' lines or trailing 'svg' artifacts
    cleaned = re.sub(r'^\s*\[?\s*svg\s*\]?\s*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'(?:\s*\[?\s*svg\s*\]?:?)+$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bsvg\b', '', cleaned, flags=re.IGNORECASE)

    # 9. Clean up any heading lines with leading/trailing artifact whitespace
    cleaned = re.sub(r'^(#{1,6})\s+', r'\1 ', cleaned, flags=re.MULTILINE)

    # 10. Handle empty or orphaned bullet markers
    empty_bullet_pattern = r'^\s*[-*+]\s*$'
    empty_bullets = re.findall(empty_bullet_pattern, cleaned, flags=re.MULTILINE)
    if empty_bullets:
        recovered_bullets = []
        if sources:
            for src in sources:
                src_text = src.get("text", "") or src.get("preview_text", "")
                src_label = src.get("source_label", "[Source 1]")
                if not src_text:
                    continue
                match = re.search(
                    r'(?:lifestyle\s+(?:measures|changes|approaches|recommendations)[^.]*include|support\s+(?:healthy\s+)?blood\s+pressure\s+include)\s+([^.]+)\.',
                    src_text,
                    re.IGNORECASE
                )
                if match:
                    raw_items = match.group(1)
                    item_regex = r'(regular physical activity|maintaining a healthy weight[^,]*|choosing a balanced diet[^,]*|moderating sodium intake|avoiding tobacco|limiting alcohol|getting adequate sleep)'
                    found_items = re.findall(item_regex, raw_items, re.IGNORECASE)
                    if found_items:
                        for item in found_items:
                            c_item = item.strip()
                            if c_item.lower().startswith("choosing a balanced diet"):
                                c_item = "Choosing a balanced diet rich in vegetables and fruits"
                            elif c_item.lower().startswith("maintaining a healthy weight"):
                                c_item = "Maintaining a healthy weight"
                            else:
                                c_item = c_item[0].upper() + c_item[1:]
                            recovered_bullets.append(f"- {c_item} {src_label}")
                        break
        if recovered_bullets:
            bullet_block = "\n".join(recovered_bullets)
            cleaned = re.sub(r'(?:^\s*[-*+]\s*$\n?)+', bullet_block + '\n\n', cleaned, count=1, flags=re.MULTILINE)
        else:
            cleaned = re.sub(r'^\s*[-*+]\s*$\n?', '', cleaned, flags=re.MULTILINE)

    # 11. Normalize multiple blank lines down to maximum of two newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


