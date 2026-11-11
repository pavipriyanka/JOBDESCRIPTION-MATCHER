"""Skill extraction using a taxonomy of canonical skills and their aliases.

"sklearn", "scikit learn" and "scikit-learn" all map to the canonical "scikit-learn",
so a resume isn't penalised for writing a skill differently from the JD.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


class SkillExtractor:
    def __init__(self, taxonomy: dict[str, dict[str, list[str]]]):
        self.category: dict[str, str] = {}
        patterns = []
        for category, skills in taxonomy.items():
            for canonical, aliases in skills.items():
                self.category[canonical] = category
                for alias in sorted(set(aliases + [canonical.lower()]), key=len, reverse=True):
                    patterns.append((canonical, self._compile(alias)))
        self.patterns = patterns

    @staticmethod
    def _compile(alias: str) -> re.Pattern:
        # Custom boundaries: letters, digits, + and # count as part of a word,
        # so "c" doesn't match inside "c++" and "r" doesn't match inside "react".
        return re.compile(r"(?<![a-z0-9+#])" + re.escape(alias.lower()) + r"(?![a-z0-9+#])")

    def extract(self, text: str) -> set[str]:
        text = text.lower()
        return {canonical for canonical, pattern in self.patterns if pattern.search(text)}

    def by_category(self, skills: set[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for skill in sorted(skills):
            grouped.setdefault(self.category[skill], []).append(skill)
        return grouped


@lru_cache(maxsize=4)
def load_extractor(path: Path) -> SkillExtractor:
    return SkillExtractor(json.loads(Path(path).read_text(encoding="utf-8")))
