"""High-level API used by the UI and the scripts."""
from __future__ import annotations

import csv
import io

from .config import Settings
from .embeddings import Encoder
from .parsing import parse_jd, parse_resume
from .scoring import MatchReport, score_resume
from .skills import load_extractor


class Matcher:
    def __init__(self, settings: Settings, encoder: Encoder):
        self.settings = settings
        self.encoder = encoder
        self.extractor = load_extractor(settings.skills_file)

    def match(self, resume_text: str, jd_text: str) -> MatchReport:
        if not jd_text.strip():
            raise ValueError("The job description is empty.")
        return score_resume(parse_resume(resume_text), parse_jd(jd_text),
                            self.extractor, self.encoder, self.settings)

    def rank(self, resumes: dict[str, str], jd_text: str) -> list[tuple[str, MatchReport]]:
        """Score many resumes against one JD, best first."""
        results = [(name, self.match(text, jd_text)) for name, text in resumes.items()]
        return sorted(results, key=lambda item: item[1].score, reverse=True)


def report_to_markdown(r: MatchReport, title: str = "Resume match report") -> str:
    lines = [
        f"# {title}", "",
        f"**Score: {r.score}/100** ({r.verdict})", "",
        f"- Skill coverage: {r.skill_score:.0%}",
        f"- Requirement coverage: {r.semantic_score:.0%}",
        f"- Resume completeness: {r.section_score:.0%}", "",
        "## Skills",
        f"- ✅ Required, matched: {', '.join(r.matched_required) or 'none'}",
        f"- ❌ Required, missing: {', '.join(r.missing_required) or 'none'}",
        f"- ✅ Preferred, matched: {', '.join(r.matched_preferred) or 'none'}",
        f"- ➖ Preferred, missing: {', '.join(r.missing_preferred) or 'none'}", "",
        "## Requirement by requirement",
        "| JD requirement | Best matching resume line | Similarity |", "|---|---|---|",
    ]
    for m in r.requirements:
        mark = "✅" if m.covered else "⚠️"
        lines.append(f"| {m.requirement} | {m.best_line} | {mark} {m.similarity:.2f} |")
    lines += ["", "## Tips"] + [f"- {t}" for t in r.tips]
    return "\n".join(lines)


def ranking_to_csv(ranking: list[tuple[str, MatchReport]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "resume", "score", "skill_coverage", "requirement_coverage",
                "completeness", "missing_required_skills"])
    for i, (name, r) in enumerate(ranking, start=1):
        w.writerow([i, name, r.score, r.skill_score, r.semantic_score, r.section_score,
                    "; ".join(r.missing_required)])
    return buf.getvalue()
