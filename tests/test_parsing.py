import io

import pytest

from matcher.parsing import parse_jd, parse_resume, read_file


def test_jd_splits_required_and_preferred(jd_text):
    jd = parse_jd(jd_text)
    assert "Strong programming skills in Python" in jd.required_text
    assert "Docker" in jd.preferred_text and "Docker" not in jd.required_text
    assert "Health insurance" not in jd.required_text + jd.preferred_text   # benefits ignored
    assert "NovaMind Labs builds" not in jd.required_text                    # company intro ignored


def test_jd_requirements_are_bullets_only(jd_text):
    reqs = parse_jd(jd_text).requirements
    assert not any("Associate AI Engineer" in r for r in reqs)
    assert reqs[0].startswith("Build and improve LLM-powered")
    assert all(not r.startswith("-") for r in reqs)


def test_jd_without_bullets_and_inline_plus():
    jd = parse_jd("We need someone who knows SQL and Excel well.\nPython is a plus.")
    assert "SQL" in jd.required_text
    assert jd.preferred_text == "Python is a plus."


def test_jd_min_years():
    assert parse_jd("Requirements:\n- 3+ years of Python experience").min_years == 3
    assert parse_jd("Requirements:\n- Python").min_years is None


def test_resume_sections_and_contact(sample_resumes):
    r = parse_resume(sample_resumes["arjun_strong_match.txt"])
    assert {"skills", "projects", "experience", "education"} <= set(r.sections)
    assert r.has_email and r.has_phone
    assert any("hybrid retrieval" in ln for ln in r.lines)
    assert not any(ln.startswith("-") for ln in r.lines)


def test_read_txt_and_docx():
    from docx import Document

    assert read_file("cv.txt", b"Python developer") == "Python developer"
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("SKILLS")
    doc.add_paragraph("Python, SQL")
    doc.save(buf)
    assert "Python, SQL" in read_file("cv.docx", buf.getvalue())


def test_read_pdf():
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "Skilled in Python and LangChain")
    c.save()
    assert "LangChain" in read_file("cv.pdf", buf.getvalue())


def test_read_file_errors():
    with pytest.raises(ValueError, match="Unsupported"):
        read_file("cv.jpg", b"x")
    with pytest.raises(ValueError, match="No text"):
        read_file("cv.txt", b"   ")
