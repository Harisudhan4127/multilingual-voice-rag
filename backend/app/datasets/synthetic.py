"""
Synthetic MSMARCO-style dataset loader.

Generates Document objects from app.datasets.synthetic_content.TOPICS.
Used when DATASET_PROVIDER=synthetic (the default, for offline dev).
"""
from __future__ import annotations

from collections.abc import Iterator

from app.datasets.base import DatasetLoader
from app.datasets.synthetic_content import TOPICS
from app.schemas.models import Document


class SyntheticDatasetLoader(DatasetLoader):
    def __init__(self, language: str = "en") -> None:
        self.language = language

    def load(self) -> Iterator[Document]:
        doc_counter = 0
        for topic, passages in TOPICS.items():
            for passage_idx, (title, text) in enumerate(passages):
                doc_counter += 1
                document_id = f"doc_{doc_counter:04d}"
                passage_id = f"{document_id}_p{passage_idx}"
                yield Document(
                    document_id=document_id,
                    passage_id=passage_id,
                    title=title,
                    text=text,
                    language=self.language,
                    metadata={"topic": topic, "source": "synthetic"},
                )
