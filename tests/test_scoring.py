import csv
import io

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from matcher import ranking_to_csv, report_to_markdown
from matcher.suggestions import rewrite_suggestions


def test_ranking_order(matcher, jd_text, sample_resumes):
    ranking = matcher.rank(sample_resumes, jd_text)
    assert [name for name, _ in ranking] == [
        "arjun_strong_match.txt", "meera_partial_match.txt", "rahul_weak_match.txt"]
    scores = [r.score for _, r in ranking]
    assert all(0 <= s <= 100 for s in scores)


def test_strong_resume_details(matcher, jd_text, sample_resumes):
    r = matcher.match(sample_resumes["arjun_strong_match.txt"], jd_text)
    assert {"Python", "LangChain", "RAG", "SQL", "Git"} <= set(r.matched_required)
    assert {"Docker", "PyTorch"} <= set(r.matched_preferred)
    assert r.section_score == 1.0
    assert len(r.requirements) == 14


def test_weak_resume_gets_useful_tips(matcher, jd_text, sample_resumes):
    r = matcher.match(sample_resumes["rahul_weak_match.txt"], jd_text)
    assert "Python" in r.missing_required
    assert {"skills", "projects", "email", "phone"} <= set(r.missing_sections)
    tips = " ".join(r.tips)
    assert "Required skills not found" in tips and "'Skills'" in tips


def test_preferred_skills_weigh_half(matcher):
    jd = "Requirements:\n- Python\nNice to have:\n- Docker"
    only_required = matcher.match("SKILLS\nPython", jd).skill_score
    only_preferred = matcher.match("SKILLS\nDocker", jd).skill_score
    assert only_required == pytest.approx(1 / 1.5, abs=1e-3)
    assert only_preferred == pytest.approx(0.5 / 1.5, abs=1e-3)


def test_empty_jd_raises(matcher):
    with pytest.raises(ValueError):
        matcher.match("Python developer", "  ")


def test_markdown_and_csv(matcher, jd_text, sample_resumes):
    ranking = matcher.rank(sample_resumes, jd_text)
    md = report_to_markdown(ranking[0][1])
    assert md.startswith("# Resume match report") and "| JD requirement |" in md
    rows = list(csv.reader(io.StringIO(ranking_to_csv(ranking))))
    assert rows[0][:3] == ["rank", "resume", "score"] and len(rows) == 4


def test_rewrite_suggestions_with_fake_llm(matcher, jd_text, sample_resumes):
    report = matcher.match(sample_resumes["meera_partial_match.txt"], jd_text)
    llm = FakeListChatModel(responses=["### Rewritten bullets\n- ..."])
    out = rewrite_suggestions(llm, sample_resumes["meera_partial_match.txt"], jd_text, report)
    assert out.startswith("### Rewritten bullets")
