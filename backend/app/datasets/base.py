"""
Dataset loader interface.

DATASET_PROVIDER=synthetic uses SyntheticDatasetLoader (Phase 2).
DATASET_PROVIDER=msmarco_xi will use MSMarcoXIDatasetLoader (Phase 10),
loading the real ai4bharat/MSMARCO-XI dataset from Hugging Face.

Both must yield the same `Document` schema so nothing downstream
(chunking, indexing, retrieval) needs to know or care which one is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.schemas.models import Document


class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Iterator[Document]:
        """Yield Document objects. Implementations may stream or load fully."""
        raise NotImplementedError

    def load_all(self) -> list[Document]:
        return list(self.load())
