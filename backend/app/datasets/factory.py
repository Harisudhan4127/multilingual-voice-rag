"""
Factory that returns the configured DatasetLoader.

DATASET_PROVIDER=synthetic   -> SyntheticDatasetLoader (offline default)
DATASET_PROVIDER=msmarco_xi  -> MSMarcoXIDatasetLoader (real HF dataset)

This is the only place that branches on DATASET_PROVIDER.
"""
from app.config import Settings
from app.datasets.base import DatasetLoader


def get_dataset_loader(settings: Settings) -> DatasetLoader:
    if settings.DATASET_PROVIDER == "synthetic":
        from app.datasets.synthetic import SyntheticDatasetLoader

        return SyntheticDatasetLoader()
    if settings.DATASET_PROVIDER == "msmarco_xi":
        from app.datasets.msmarco_xi import MSMarcoXIDatasetLoader

        return MSMarcoXIDatasetLoader(
            language=settings.DATASET_LANGUAGE,
            split=settings.MSMARCO_SPLIT,
            max_documents=settings.MSMARCO_MAX_DOCS,
        )
    raise ValueError(f"Unknown DATASET_PROVIDER: {settings.DATASET_PROVIDER}")
