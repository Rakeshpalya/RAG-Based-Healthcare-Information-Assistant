"""
Unit and Integration Tests for AI Healthcare Agent Response Rendering & Markdown Sanitization.
Validates:
1. Removes [svg](http://localhost:8501/...) and [svg](#...) artifacts.
2. Removes localhost anchor links, raw SVGs, and heading anchor tags.
3. Preserves legitimate clinical content:
   - Headings (### Title)
   - Bullet lists (- item)
   - Bold text (**text**)
   - Inline citations ([Source 1])
   - Medical disclaimers & Emergency advisories
4. Generic handling across multiple diverse medical topics (Hypertension, Asthma, Cardiovascular Risk).
5. No hard-coding for 'hypertension' or specific document names.
"""

import pytest
from backend.rag.markdown_utils import clean_ai_markdown as backend_clean_markdown
from frontend.utils.helpers import clean_ai_markdown as frontend_clean_markdown


SAMPLE_HYPERTENSION_RAW = """
[svg](http://localhost:8501/#what-is-hypertension)

### What Is Hypertension?

Hypertension, commonly known as high blood pressure, is a chronic clinical condition where arterial pressure is persistently elevated [Source 1].

[svg](http://localhost:8501/#common-risk-factors)

### Common Risk Factors

- Increasing age
- Family history of cardiovascular disease
- Excess body weight and obesity
- High dietary sodium consumption

[svg](http://localhost:8501/#recommended-lifestyle-measures)

### Recommended Lifestyle Measures

- Regular aerobic physical activity
- Balanced dietary pattern such as DASH
- Moderating sodium intake

[Source 1]

MEDICAL DISCLAIMER: This assistant provides research information and does not replace professional clinical diagnosis.
"""

SAMPLE_ASTHMA_RAW = """
### [svg](http://localhost:8501/#clinical-symptoms-of-asthma) Clinical Symptoms of Asthma

Acute asthma exacerbation presents with progressive wheezing, shortness of breath, and chest tightness [Source 2].

<a class="anchor-link" href="#emergency-advisory"><svg viewBox="0 0 16 16"><path d="..."></path></svg></a>
### Emergency Advisory

If the patient experiences severe dyspnea, cyanosis, or silent chest, seek immediate emergency medical care!

[Source 2]
"""

SAMPLE_CARDIO_RAW = """
### Cardiovascular Risk Assessment [svg](http://localhost:8501/#cardiovascular-risk-assessment)

Key indicators for cardiovascular risk stratify according to:
- Systolic and diastolic blood pressure levels
- Lipid panel measurements (LDL, HDL, triglycerides)
- Smoking status and glycemic control [Source 1, Source 3]

<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"></svg>
http://localhost:8501/#staging-considerations
### Staging Considerations

Stage 1 vs Stage 2 thresholds guide pharmacological intervention.
"""


@pytest.mark.parametrize("clean_fn", [backend_clean_markdown, frontend_clean_markdown])
def test_clean_markdown_hypertension_question(clean_fn):
    """Verifies that the hypertension question response has all [svg] artifacts stripped and all headings intact."""
    result = clean_fn(SAMPLE_HYPERTENSION_RAW)

    # 1. Verify all unwanted artifacts are removed
    assert "[svg]" not in result.lower()
    assert "localhost:8501" not in result
    assert "<svg" not in result
    assert "<a" not in result

    # 2. Verify headings are rendered normally
    assert "### What Is Hypertension?" in result
    assert "### Common Risk Factors" in result
    assert "### Recommended Lifestyle Measures" in result

    # 3. Verify legitimate citations are preserved
    assert "[Source 1]" in result

    # 4. Verify bullet lists and clinical text are preserved
    assert "- Increasing age" in result
    assert "- Regular aerobic physical activity" in result
    assert "MEDICAL DISCLAIMER:" in result


@pytest.mark.parametrize("clean_fn", [backend_clean_markdown, frontend_clean_markdown])
def test_clean_markdown_asthma_question(clean_fn):
    """Verifies that asthma question with inline and raw HTML anchor artifacts is cleaned properly."""
    result = clean_fn(SAMPLE_ASTHMA_RAW)

    assert "[svg]" not in result.lower()
    assert "localhost" not in result
    assert "anchor-link" not in result
    assert "<svg" not in result

    # Preserves heading and emergency advisory
    assert "### Clinical Symptoms of Asthma" in result
    assert "### Emergency Advisory" in result
    assert "acute asthma exacerbation presents with progressive wheezing" in result.lower()
    assert "[Source 2]" in result


@pytest.mark.parametrize("clean_fn", [backend_clean_markdown, frontend_clean_markdown])
def test_clean_markdown_cardiovascular_question(clean_fn):
    """Verifies that cardiovascular risk question with standalone links and svg tags is sanitized."""
    result = clean_fn(SAMPLE_CARDIO_RAW)

    assert "[svg]" not in result.lower()
    assert "localhost" not in result
    assert "<svg" not in result
    assert "http://" not in result

    assert "### Cardiovascular Risk Assessment" in result
    assert "### Staging Considerations" in result
    assert "- Systolic and diastolic blood pressure levels" in result
    assert "[Source 1, Source 3]" in result


def test_clean_markdown_preserves_legitimate_content():
    """Verifies edge cases like empty string, normal text without artifacts, complex citations."""
    assert backend_clean_markdown("") == ""
    assert backend_clean_markdown(None) == ""

    normal_text = "This is a normal paragraph with **bold** text and [Source 1, 2]."
    assert backend_clean_markdown(normal_text) == normal_text


@pytest.mark.parametrize("clean_fn", [backend_clean_markdown, frontend_clean_markdown])
def test_clean_markdown_bottom_artifacts(clean_fn):
    """Verifies that [svg], reference link definitions, and trailing svg artifacts at the bottom are stripped."""
    # Case 1: Trailing [svg] at the very bottom
    t1 = "Clinical answer discussing hypertension risk factors [Source 1].\n\n[svg]"
    res1 = clean_fn(t1)
    assert "svg" not in res1.lower()
    assert "[Source 1]" in res1

    # Case 2: Reference link definition at bottom
    t2 = "Clinical answer [Source 1].\n\n[svg]: http://localhost:8501/#recommended-lifestyle-measures"
    res2 = clean_fn(t2)
    assert "svg" not in res2.lower()
    assert "localhost" not in res2

    # Case 3: Anchor reference link at bottom
    t3 = "Clinical answer [Source 1].\n\n[svg]: #lifestyle-changes"
    res3 = clean_fn(t3)
    assert "svg" not in res3.lower()

    # Case 4: Standalone svg text on its own line
    t4 = "Clinical answer [Source 1].\n\nsvg\n"
    res4 = clean_fn(t4)
    assert "svg" not in res4.lower()

    # Case 5: Space between bracket and parenthesis
    t5 = "Clinical answer [Source 1].\n\n[svg] (http://localhost:8501/#bottom)"
    res5 = clean_fn(t5)
    assert "svg" not in res5.lower()
    assert "localhost" not in res5

    # Case 6: Modern Streamlit aria-label anchor tag
    t6 = "### What Is Hypertension?\n<a aria-label=\"Link to heading\" href=\"#what-is-hypertension\"><svg viewBox=\"0 0 16 16\"></svg></a>\nText [Source 1]."
    res6 = clean_fn(t6)
    assert "svg" not in res6.lower()
    assert "<a" not in res6
    assert "### What Is Hypertension?" in res6
    assert "[Source 1]" in res6

