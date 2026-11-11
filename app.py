"""Streamlit UI for the Resume–JD Matcher.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from matcher import Matcher, ranking_to_csv, read_file, report_to_markdown, settings  # noqa: E402
from matcher.embeddings import SentenceTransformerEncoder  # noqa: E402
from matcher.suggestions import get_llm, rewrite_suggestions  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
st.set_page_config(page_title="Resume–JD Matcher", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_matcher() -> Matcher:
    return Matcher(settings, SentenceTransformerEncoder(settings.embedding_model))


def jd_input(key: str) -> str:
    if st.button("Load sample JD", key=f"{key}_sample"):
        st.session_state[f"{key}_jd"] = (SAMPLES / "jd_ai_engineer.txt").read_text(encoding="utf-8")
    return st.text_area("Job description", height=320, key=f"{key}_jd",
                        placeholder="Paste the full job description here...")


def chips(label: str, items: list[str], color: str):
    st.markdown(f"**{label}**")
    st.markdown(" ".join(f":{color}-background[{s}]" for s in items) if items else "_none_")


def show_report(report, resume_text: str, jd_text: str, api_key: str):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match score", f"{report.score}/100")
    c2.metric("Skill coverage", f"{report.skill_score:.0%}")
    c3.metric("Requirement coverage", f"{report.semantic_score:.0%}")
    c4.metric("Completeness", f"{report.section_score:.0%}")
    st.progress(report.score / 100, text=report.verdict)

    left, right = st.columns(2)
    with left:
        chips("✅ Required skills you have", report.matched_required, "green")
        chips("❌ Required skills missing", report.missing_required, "red")
    with right:
        chips("✅ Nice-to-have skills you have", report.matched_preferred, "green")
        chips("➖ Nice-to-have skills missing", report.missing_preferred, "orange")
    if report.extra_skills:
        chips("Other skills on your resume", report.extra_skills, "gray")

    st.subheader("💡 Tips")
    for tip in report.tips or ["Looks great. Tailor your summary to the company and apply!"]:
        st.markdown(f"- {tip}")

    st.subheader("📋 Requirement by requirement")
    st.dataframe(pd.DataFrame([{
        "": "✅" if m.covered else "⚠️", "JD requirement": m.requirement,
        "Best matching resume line": m.best_line, "Similarity": m.similarity,
    } for m in report.requirements]), hide_index=True, width="stretch")

    st.subheader("✍️ AI rewrite suggestions")
    if st.button("Suggest improved bullets (Gemini)", disabled=not api_key,
                 help=None if api_key else "Add a Gemini API key in the sidebar"):
        with st.spinner("Asking Gemini..."):
            try:
                st.markdown(rewrite_suggestions(get_llm(settings, api_key), resume_text, jd_text, report))
            except Exception as e:  # bad key, quota, network
                st.error(f"LLM request failed: {e}")

    st.download_button("⬇️ Download report (Markdown)", report_to_markdown(report),
                       file_name="match_report.md", mime="text/markdown")


def single_tab(api_key: str):
    col1, col2 = st.columns(2)
    with col1:
        up = st.file_uploader("Resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        pasted = st.text_area("…or paste resume text", height=230, disabled=up is not None)
    with col2:
        jd_text = jd_input("single")

    if st.button("Analyse match", type="primary"):
        try:
            resume_text = read_file(up.name, up.getvalue()) if up else pasted
            if not resume_text.strip():
                st.warning("Upload or paste a resume first.")
                return
            st.session_state.single = (load_matcher().match(resume_text, jd_text), resume_text, jd_text)
        except ValueError as e:
            st.error(str(e))
            return
    if "single" in st.session_state:
        show_report(*st.session_state.single, api_key)


def rank_tab():
    st.caption("Recruiter view: rank several resumes against one job description.")
    col1, col2 = st.columns(2)
    with col1:
        ups = st.file_uploader("Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        use_samples = st.checkbox("Use the 3 sample resumes")
    with col2:
        jd_text = jd_input("rank")

    if st.button("Rank resumes", type="primary"):
        resumes = {}
        try:
            for f in ups or []:
                resumes[f.name] = read_file(f.name, f.getvalue())
            if use_samples:
                for p in sorted((SAMPLES / "resumes").glob("*.txt")):
                    resumes[p.name] = p.read_text(encoding="utf-8")
            if not resumes:
                st.warning("Upload resumes or tick 'Use the 3 sample resumes'.")
                return
            ranking = load_matcher().rank(resumes, jd_text)
        except ValueError as e:
            st.error(str(e))
            return
        st.dataframe(pd.DataFrame([{
            "Rank": i, "Resume": name, "Score": r.score, "Verdict": r.verdict,
            "Missing required skills": ", ".join(r.missing_required) or "–",
        } for i, (name, r) in enumerate(ranking, start=1)]), hide_index=True, width="stretch")
        st.download_button("⬇️ Download ranking (CSV)", ranking_to_csv(ranking),
                           file_name="ranking.csv", mime="text/csv")


def main():
    st.title("🎯 Resume vs Job Description Matcher")
    st.caption("Skill taxonomy with synonyms · required vs nice-to-have · semantic requirement matching")
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API key (optional)", value=settings.google_api_key, type="password",
                                help="Only needed for AI rewrite suggestions. Scoring works without it.")
        st.markdown("**How the score works**")
        st.caption(f"{settings.w_skills:.0%} skill coverage + {settings.w_semantic:.0%} requirement coverage "
                   f"+ {settings.w_sections:.0%} resume completeness")

    tab1, tab2 = st.tabs(["📄 Check my resume", "🏆 Rank resumes"])
    with tab1:
        single_tab(api_key)
    with tab2:
        rank_tab()


main()
