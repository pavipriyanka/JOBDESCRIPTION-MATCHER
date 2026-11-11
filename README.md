# 🎯 Resume vs Job Description Matcher

An ATS-style tool that scores how well a resume fits a job description and explains **why**: which required skills are missing, which JD requirements your resume doesn't address, and what to fix. It can also **rank many resumes** for one job (recruiter view) and suggest **truthful AI rewrites** of resume bullets with Gemini.

## Features
- 📄 Reads **PDF, DOCX and TXT** resumes
- 🧠 **Skill taxonomy with synonyms**: 75 skills in 9 categories; `sklearn` = `scikit-learn`, `FAISS` → Vector Databases
- ⚖️ **Required vs nice-to-have**: the JD is split by its headings ("Requirements", "Nice to have") and inline cues ("…is a plus"); preferred skills count half
- 🔍 **Requirement-by-requirement matching**: each JD bullet is matched to its most similar resume line with sentence embeddings
- 🧾 **Resume completeness check**: Skills / Projects / Education sections, email, phone
- 💡 **Actionable tips**: missing skills, uncovered requirements, bullets without numbers, experience gaps
- ✍️ **AI rewrite suggestions** (optional, Gemini) with a prompt that forbids inventing skills
- 🏆 **Batch ranking** of resumes with CSV export
- ✅ **20 offline unit tests**

## How the score works
```
score = 50% × skill coverage        (required skills ×1, nice-to-have ×0.5)
      + 40% × requirement coverage  (mean of rescaled best-match cosine per JD requirement)
      + 10% × resume completeness   (skills, projects, education, email, phone)
```
| Score | Verdict |
|---|---|
| 75+ | Strong match: apply now |
| 55–74 | Good match: tailor a few points, then apply |
| 35–54 | Partial match: close the skill gaps first |
| < 35 | Weak match |

## Architecture
```
resume (PDF/DOCX/TXT) ──► parse_resume ──► sections + bullet lines + contact info
job description       ──► parse_jd     ──► required text · preferred text · requirement bullets · min years
                                   │
          ┌────────────────────────┼─────────────────────────────┐
          ▼                        ▼                             ▼
  SkillExtractor (taxonomy,   Sentence embeddings:          Section check
  aliases, regex boundaries)  JD bullets × resume lines
          │                   cosine matrix → best match
          └────────────► weighted score + tips ◄─────────────────┘
                                   │
                    optional: Gemini rewrite suggestions
```

## Project structure
```
resume-jd-matcher/
├── app.py                    # Streamlit UI (2 tabs: check / rank)
├── src/matcher/
│   ├── config.py             # weights, thresholds, model names
│   ├── parsing.py            # file reading, resume sections, JD required/preferred split
│   ├── skills.py             # taxonomy-based skill extraction
│   ├── embeddings.py         # sentence-transformers encoder (+ offline encoder for tests)
│   ├── scoring.py            # scoring, verdict, tips
│   ├── suggestions.py        # Gemini rewrite suggestions
│   └── service.py            # Matcher API, Markdown/CSV export
├── scripts/
│   ├── score.py              # score one resume from the terminal
│   └── rank.py               # rank a folder of resumes → CSV
├── data/
│   ├── skills.json           # editable skill taxonomy
│   └── samples/              # sample JD + 3 resumes (strong / partial / weak)
└── tests/                    # pytest suite
```

## Quick start
Python 3.11 or 3.12 recommended (3.10+ works).
```bash
python -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # optional: add GOOGLE_API_KEY for AI rewrites
streamlit run app.py
```
Click **Load sample JD**, upload your resume (or paste it) and click **Analyse match**. On the **Rank resumes** tab, tick *Use the 3 sample resumes* to see ranking. The first run downloads the embedding model (~90 MB) once.

### Terminal
```bash
python scripts/score.py data/samples/resumes/arjun_strong_match.txt data/samples/jd_ai_engineer.txt
python scripts/rank.py data/samples/resumes data/samples/jd_ai_engineer.txt --out ranking.csv
```

### Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Key design decisions
| Decision | Why |
|---|---|
| Skill taxonomy with aliases | Exact keyword matching misses `sklearn` vs `scikit-learn`; a taxonomy normalises both sides |
| Custom regex boundaries (`+`, `#` count as word characters) | Stops `C` matching inside `C++` and `R` matching inside `React` |
| Required vs preferred weighting | Missing a "nice to have" shouldn't hurt as much as missing a requirement |
| Per-requirement matching instead of whole-document similarity | A whole-document embedding blurs everything together; per-requirement matching shows *which* requirement is uncovered |
| Rescaling cosine (0.15–0.75 → 0–1) | MiniLM cosines rarely use the full range, so raw values would compress all scores together |
| Only JD bullets as requirements (when present) | Skips titles and company intro text that aren't requirements |
| LLM prompt forbids inventing facts, uses `[add number]` placeholders | Keeps suggestions truthful, which matters in interviews |
| Encoder as a Protocol + offline hashing encoder | Tests run in under a second with no model download |

## Limitations & future work
- Skills not in `skills.json` aren't detected; add them there (it's plain JSON)
- Score weights are hand-set; they could be calibrated on resumes labelled by recruiters
- Section detection relies on common heading names
- Next: cross-encoder re-ranking for requirement matching, LLM-based skill extraction for unknown skills, OCR for scanned resumes

## Tech stack
Python · sentence-transformers · NumPy · pypdf · python-docx · LangChain + Gemini · Streamlit · pandas · pytest
