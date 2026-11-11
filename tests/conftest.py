import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matcher import Matcher, settings  # noqa: E402
from matcher.embeddings import HashingEncoder  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


@pytest.fixture(scope="session")
def matcher() -> Matcher:
    # Offline encoder: no model download needed for tests.
    return Matcher(settings, HashingEncoder())


@pytest.fixture(scope="session")
def jd_text() -> str:
    return (SAMPLES / "jd_ai_engineer.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def sample_resumes() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((SAMPLES / "resumes").glob("*.txt"))}
