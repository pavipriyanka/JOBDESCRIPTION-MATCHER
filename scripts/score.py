"""Score one resume against one job description from the terminal.

Usage:
  python scripts/score.py data/samples/resumes/arjun_strong_match.txt data/samples/jd_ai_engineer.txt
  python scripts/score.py my_resume.pdf jd.txt --out report.md
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matcher import Matcher, read_file, report_to_markdown, settings  # noqa: E402
from matcher.embeddings import SentenceTransformerEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resume", type=Path, help="PDF, DOCX or TXT")
    ap.add_argument("jd", type=Path, help="job description as a text file")
    ap.add_argument("--out", type=Path, help="save the Markdown report to this file")
    args = ap.parse_args()

    matcher = Matcher(settings, SentenceTransformerEncoder(settings.embedding_model))
    report = matcher.match(read_file(args.resume.name, args.resume.read_bytes()),
                           args.jd.read_text(encoding="utf-8"))
    md = report_to_markdown(report, title=f"{args.resume.name} vs {args.jd.name}")
    print(md)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
