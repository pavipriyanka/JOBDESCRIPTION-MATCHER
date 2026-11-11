"""Rank every resume in a folder against one job description and save a CSV.

Usage:
  python scripts/rank.py data/samples/resumes data/samples/jd_ai_engineer.txt --out ranking.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matcher import Matcher, ranking_to_csv, read_file, settings  # noqa: E402
from matcher.embeddings import SentenceTransformerEncoder  # noqa: E402

EXTENSIONS = {".pdf", ".docx", ".txt"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder of resumes (PDF, DOCX, TXT)")
    ap.add_argument("jd", type=Path, help="job description as a text file")
    ap.add_argument("--out", type=Path, default=Path("ranking.csv"))
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit(f"No PDF/DOCX/TXT files found in {args.folder}")
    resumes = {p.name: read_file(p.name, p.read_bytes()) for p in files}

    matcher = Matcher(settings, SentenceTransformerEncoder(settings.embedding_model))
    ranking = matcher.rank(resumes, args.jd.read_text(encoding="utf-8"))

    print(f"{'#':<3} {'score':>5}  resume")
    for i, (name, r) in enumerate(ranking, start=1):
        print(f"{i:<3} {r.score:>5}  {name}  ({r.verdict})")
    args.out.write_text(ranking_to_csv(ranking), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
