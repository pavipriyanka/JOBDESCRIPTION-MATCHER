"""Configuration from environment variables / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    skills_file: Path = DATA_DIR / "skills.json"

    # Final score weights (must sum to 1)
    w_skills: float = 0.5
    w_semantic: float = 0.4
    w_sections: float = 0.1

    # Raw cosine similarities from MiniLM rarely go below ~0.15 or above ~0.75,
    # so they are rescaled linearly from this range to 0..1.
    sim_floor: float = 0.15
    sim_ceiling: float = 0.75
    requirement_match_threshold: float = 0.45  # raw cosine to count a requirement as covered
    preferred_weight: float = 0.5               # nice-to-have skills count half


settings = Settings()
