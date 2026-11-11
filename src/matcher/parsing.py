"""Read resumes (PDF / DOCX / TXT) and split resumes and job descriptions into parts."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

RESUME_SECTIONS = {
    "summary": ["summary", "professional summary", "objective", "profile", "about me"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "tools", "technologies"],
    "experience": ["experience", "work experience", "internship", "internships", "employment", "professional experience"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "education": ["education", "academic background", "academics", "qualifications"],
    "certifications": ["certifications", "certificates", "courses", "training"],
    "achievements": ["achievements", "awards", "accomplishments", "extracurricular", "activities"],
}
REQUIRED_SECTIONS = ["skills", "projects", "education"]

_PREFERRED = re.compile(r"\b(preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|plus|desirable|optional)\b", re.I)
_REQUIRED = re.compile(r"\b(requirements?|required|must[- ]have|qualifications|responsibilities|what you.?ll do|"
                       r"what we.?re looking for|skills|you have|key skills|eligibility)\b", re.I)
_HEADING_START = re.compile(r"^(about|benefits|perks|what we offer|location|salary|requirements?|required skills|qualifications|responsibilities|preferred|"
                            r"nice[- ]to[- ]have|good[- ]to[- ]have|bonus|skills|key skills|what you|about the role|eligibility)\b.{0,25}$", re.I)
_IGNORE = re.compile(r"\b(about (us|the company|company)|who we are|benefits|perks|what we offer|salary|"
                     r"compensation|ctc|location|how to apply|equal opportunity)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-•*▪●◦‣]|\d+[.)])\s*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?:\+?\d[\d\s-]{8,}\d)")
_YEARS = re.compile(r"(\d+)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)", re.I)


# ---------------- file reading ----------------
def read_file(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    elif ext == "docx":
        from docx import Document

        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    elif ext in {"txt", "md"}:
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: .{ext} (use PDF, DOCX or TXT)")
    if not text.strip():
        raise ValueError(f"No text found in {name}. Scanned PDFs need OCR.")
    return text


# ---------------- resume ----------------
@dataclass
class Resume:
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)   # meaningful lines/bullets for semantic matching
    has_email: bool = False
    has_phone: bool = False


def _heading_of(line: str) -> str | None:
    clean = re.sub(r"[^a-z ]", "", line.lower()).strip()
    if not clean or len(clean.split()) > 4:
        return None
    for section, names in RESUME_SECTIONS.items():
        if clean in names:
            return section
    return None


def parse_resume(text: str) -> Resume:
    sections: dict[str, list[str]] = {}
    current = "header"
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _heading_of(line)
        if heading:
            current = heading
            continue
        sections.setdefault(current, []).append(line)
        content = _BULLET.sub("", line)
        if len(content.split()) >= 4:          # skip fragments like phone numbers or dates
            lines.append(content)
    return Resume(
        text=text,
        sections={k: "\n".join(v) for k, v in sections.items()},
        lines=lines,
        has_email=bool(_EMAIL.search(text)),
        has_phone=bool(_PHONE.search(text)),
    )


# ---------------- job description ----------------
@dataclass
class JobDescription:
    text: str
    required_text: str
    preferred_text: str
    requirements: list[str]       # individual requirement lines/bullets
    min_years: int | None


def parse_jd(text: str) -> JobDescription:
    """Split a JD into required vs preferred parts by following its headings."""
    mode = "required"
    required, preferred, requirements = [], [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_heading = (not _BULLET.match(line) and len(line) < 60
                      and (line.endswith(":") or line.isupper() or bool(_HEADING_START.match(line))))
        if is_heading:
            if _IGNORE.search(line):
                mode = "ignore"            # company intro, benefits, salary... not requirements
            elif _PREFERRED.search(line):
                mode = "preferred"
            elif _REQUIRED.search(line):
                mode = "required"
            continue
        if mode == "ignore":
            continue
        # a single line like "Experience with Docker is a plus" is preferred on its own
        line_mode = "preferred" if _PREFERRED.search(line) else mode
        (preferred if line_mode == "preferred" else required).append(line)
        content = _BULLET.sub("", line)
        if len(content.split()) >= 3:
            requirements.append((bool(_BULLET.match(line)), content))
    # If the JD uses bullet points, only bullets are requirements (skips titles and intro prose).
    if any(is_bullet for is_bullet, _ in requirements):
        requirements = [(b, c) for b, c in requirements if b]
    years = [int(y) for y in _YEARS.findall(text)]
    return JobDescription(
        text=text,
        required_text="\n".join(required),
        preferred_text="\n".join(preferred),
        requirements=[c for _, c in requirements],
        min_years=min(years) if years else None,
    )
