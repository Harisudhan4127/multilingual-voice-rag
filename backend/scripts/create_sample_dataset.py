"""
Generate the synthetic MSMARCO-style dataset and write it to data/raw/.

Usage:
    python scripts/create_sample_dataset.py
    python scripts/create_sample_dataset.py --out data/raw/custom.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.datasets.synthetic import SyntheticDatasetLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/synthetic_dataset.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = SyntheticDatasetLoader()
    documents = loader.load_all()

    with out_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}")

    topics = sorted({doc.metadata.get("topic", "unknown") for doc in documents})
    print(f"Topics ({len(topics)}): {', '.join(topics)}")


if __name__ == "__main__":
    main()
