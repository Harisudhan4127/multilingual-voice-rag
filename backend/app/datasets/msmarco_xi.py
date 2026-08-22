"""
Real MSMARCO-XI dataset adapter (Phase 10 / Stage 2 of the dev strategy).

Loads ai4bharat/MSMARCO-XI (MS MARCO translated into 14 Indic languages)
from the Hugging Face Hub and yields the same `Document` schema as the
synthetic loader, so chunking/indexing/retrieval need zero changes when
DATASET_PROVIDER=msmarco_xi.

Row schema (verified against the dataset's builder script):
    query, Answer, query_id, query_type,
    passages: {is_selected: [0/1], English_passages: [...], Translated_passages: [...]},
    Eng_Query, ...

Notes:
  - The dataset has no passage titles; we synthesize short titles from the
    query so citations remain readable.
  - `train` holds millions of rows; default split is `validation` and the
    loader caps documents at MSMARCO_MAX_DOCS.
  - Requires the optional `datasets` package (pip install datasets).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from app.datasets.base import DatasetLoader
from app.schemas.models import Document

logger = logging.getLogger("voice_rag")

_HF_DATASET = "ai4bharat/MSMARCO-XI"


class MSMarcoXIDatasetLoader(DatasetLoader):
    def __init__(
        self,
        language: str = "hi",
        split: str = "validation",
        max_documents: int = 500,
        use_english_passages: bool = False,
    ) -> None:
        self.language = language
        self.split = split
        self.max_documents = max_documents
        self.use_english_passages = use_english_passages

    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_documents(row: dict[str, Any], language: str, use_english: bool) -> list[Document]:
        """Pure mapping logic -- unit-testable without network or HF."""
        query_id = row.get("query_id")
        passages = row.get("passages", {}) or {}
        is_selected = passages.get("is_selected", []) or []
        english = passages.get("English_passages", []) or []
        translated = passages.get("Translated_passages", []) or []

        texts_pool = english if use_english else translated
        if not texts_pool:
            texts_pool = english  # fall back to English if translations missing

        docs: list[Document] = []
        for i, text in enumerate(texts_pool):
            text = (text or "").strip()
            if not text:
                continue
            selected = bool(is_selected[i]) if i < len(is_selected) else False
            title_source = row.get("Eng_Query") or row.get("query") or ""
            title = " ".join(title_source.split())[:80] or f"Passage {i}"
            docs.append(
                Document(
                    document_id=f"msmarco_{query_id}_{i}",
                    passage_id=f"msmarco_{query_id}_p{i}",
                    title=title,
                    text=text,
                    language=language,
                    metadata={
                        "source": "msmarco_xi",
                        "query_id": query_id,
                        "query": row.get("query"),
                        "eng_query": row.get("Eng_Query"),
                        "passage_index": i,
                        "is_selected": selected,
                    },
                )
            )

        if docs:
            # Stable, meaningful order: selected passages first.
            docs.sort(key=lambda d: not d.metadata["is_selected"])
        return docs

    # ------------------------------------------------------------------ #

    def _rows(self) -> Iterator[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "DATASET_PROVIDER=msmarco_xi requires the `datasets` package: "
                "pip install datasets"
            ) from exc

        logger.info(
            "Streaming MSMARCO-XI config=%s split=%s", self.language, self.split
        )
        ds = load_dataset(_HF_DATASET, self.language, split=self.split, streaming=True)
        yield from ds

    def load(self) -> Iterator[Document]:
        count = 0
        for row in self._rows():
            for doc in self._row_to_documents(row, self.language, self.use_english_passages):
                yield doc
                count += 1
                if count >= self.max_documents:
                    return
