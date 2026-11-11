"""Scoring a resume against a job description.

Final score (0-100) =
    w_skills   * weighted skill coverage   (required skills count 1, preferred 0.5)
  + w_semantic * requirement coverage      (each JD requirement vs its best resume line)
  + w_sections * resume completeness       (skills/projects/education sections, email, phone)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .embeddings import Encoder, cosine_matrix
from .parsing import REQUIRED_SECTIONS, JobDescription, Resume
from .skills import SkillExtractor


@dataclass
class RequirementMatch:
    requirement: str
    best_line: str
    similarity: float
    covered: bool


@dataclass
class MatchReport:
    score: int
    skill_score: float
    semantic_score: float
    section_score: float
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]
    extra_skills: list[str]                       # resume skills the JD didn't ask for
    requirements: list[RequirementMatch]
    missing_sections: list[str]
    jd_min_years: int | None
    tips: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.score >= 75:
            return "Strong match: apply now"
        if self.score >= 55:
            return "Good match: tailor a few points, then apply"
        if self.score >= 35:
            return "Partial match: close the skill gaps first"
        return "Weak match: probably not the right role"


def _rescale(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def skill_coverage(jd: JobDescription, resume: Resume, extractor: SkillExtractor, preferred_weight: float):
    required = extractor.extract(jd.required_text)
    preferred = extractor.extract(jd.preferred_text) - required
    have = extractor.extract(resume.text)
    total = len(required) + preferred_weight * len(preferred)
    got = len(required & have) + preferred_weight * len(preferred & have)
    score = got / total if total else 0.0
    return score, required, preferred, have


def requirement_coverage(jd: JobDescription, resume: Resume, encoder: Encoder, s: Settings):
    if not jd.requirements or not resume.lines:
        return 0.0, []
    sims = cosine_matrix(encoder.encode(jd.requirements), encoder.encode(resume.lines))
    best = sims.argmax(axis=1)
    matches = [
        RequirementMatch(req, resume.lines[j], round(float(sims[i, j]), 3),
                         float(sims[i, j]) >= s.requirement_match_threshold)
        for i, (req, j) in enumerate(zip(jd.requirements, best))
    ]
    score = float(np.mean([_rescale(m.similarity, s.sim_floor, s.sim_ceiling) for m in matches]))
    return score, matches


def section_completeness(resume: Resume):
    missing = [sec for sec in REQUIRED_SECTIONS if sec not in resume.sections]
    if not resume.has_email:
        missing.append("email")
    if not resume.has_phone:
        missing.append("phone")
    checks = len(REQUIRED_SECTIONS) + 2
    return (checks - len(missing)) / checks, missing


def build_tips(report: MatchReport, resume: Resume) -> list[str]:
    tips = []
    if report.missing_required:
        tips.append("Required skills not found: " + ", ".join(report.missing_required[:8]) +
                    ". If you have them, add them to Skills AND show them in a project bullet.")
    if report.missing_preferred:
        tips.append("Nice-to-have skills you could highlight: " + ", ".join(report.missing_preferred[:6]) + ".")
    weak = [m for m in report.requirements if not m.covered]
    if weak:
        tips.append(f"{len(weak)} of {len(report.requirements)} JD requirements have no clearly matching "
                    f"resume line, e.g. \"{weak[0].requirement[:90]}\". Add a bullet that addresses it directly.")
    unquantified = [ln for ln in resume.lines if not any(ch.isdigit() for ch in ln)]
    if resume.lines and len(unquantified) / len(resume.lines) > 0.7:
        tips.append("Most bullets have no numbers. Add results such as accuracy, users, time saved or dataset size.")
    for sec in report.missing_sections:
        tips.append(f"Add a clear '{sec.title()}' section/heading so ATS software can find it.")
    if report.jd_min_years and report.jd_min_years >= 2:
        tips.append(f"The JD asks for {report.jd_min_years}+ years of experience. As a fresher, lead with "
                    "internships and substantial projects, and still apply if the skills match.")
    return tips


def score_resume(resume: Resume, jd: JobDescription, extractor: SkillExtractor,
                 encoder: Encoder, s: Settings) -> MatchReport:
    skill_score, required, preferred, have = skill_coverage(jd, resume, extractor, s.preferred_weight)
    semantic_score, matches = requirement_coverage(jd, resume, encoder, s)
    section_score, missing_sections = section_completeness(resume)
    total = s.w_skills * skill_score + s.w_semantic * semantic_score + s.w_sections * section_score
    report = MatchReport(
        score=round(100 * total),
        skill_score=round(skill_score, 3),
        semantic_score=round(semantic_score, 3),
        section_score=round(section_score, 3),
        matched_required=sorted(required & have),
        missing_required=sorted(required - have),
        matched_preferred=sorted(preferred & have),
        missing_preferred=sorted(preferred - have),
        extra_skills=sorted(have - required - preferred),
        requirements=matches,
        missing_sections=missing_sections,
        jd_min_years=jd.min_years,
    )
    report.tips = build_tips(report, resume)
    return report
