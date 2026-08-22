"""
Hand-labeled query -> relevant document_id(s) pairs for the synthetic
dataset, used by the chunking evaluation (Section 8) and later the
retrieval evaluation (Section 27).

document_id assignment is deterministic (see SyntheticDatasetLoader):
topics are iterated in TOPICS dict order, and passages within each topic
in list order, numbered doc_0001, doc_0002, ... continuing across topics.
IDs below were derived by inspecting data/raw/synthetic_dataset.jsonl,
not guessed.
"""
from pydantic import BaseModel


class EvalQuery(BaseModel):
    query: str
    relevant_document_ids: list[str]


EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery(
        query="What is supervised learning?",
        relevant_document_ids=["doc_0001"],
    ),
    EvalQuery(
        query="How do you prevent overfitting in a machine learning model?",
        relevant_document_ids=["doc_0002"],
    ),
    EvalQuery(
        query="What are convolutional neural networks used for?",
        relevant_document_ids=["doc_0003"],
    ),
    EvalQuery(
        query="Explain gradient descent and the Adam optimizer.",
        relevant_document_ids=["doc_0004"],
    ),
    EvalQuery(
        query="What were the main cities of the Indus Valley Civilization?",
        relevant_document_ids=["doc_0005"],
    ),
    EvalQuery(
        query="Why did Ashoka convert to Buddhism?",
        relevant_document_ids=["doc_0006"],
    ),
    EvalQuery(
        query="Who built the Taj Mahal and why?",
        relevant_document_ids=["doc_0007"],
    ),
    EvalQuery(
        query="What role did Mahatma Gandhi play in the independence movement?",
        relevant_document_ids=["doc_0008"],
    ),
    EvalQuery(
        query="How do greenhouse gases trap heat in the atmosphere?",
        relevant_document_ids=["doc_0009"],
    ),
    EvalQuery(
        query="How does carbon dioxide absorption affect ocean pH?",
        relevant_document_ids=["doc_0010"],
    ),
    EvalQuery(
        query="Why has solar and wind energy become cheaper?",
        relevant_document_ids=["doc_0011"],
    ),
    EvalQuery(
        query="How does blood flow between the heart and lungs?",
        relevant_document_ids=["doc_0012"],
    ),
    EvalQuery(
        query="What is the difference between innate and adaptive immunity?",
        relevant_document_ids=["doc_0013"],
    ),
    EvalQuery(
        query="What part of the brain controls the sleep-wake cycle?",
        relevant_document_ids=["doc_0014"],
    ),
    EvalQuery(
        query="What is the difference between a git commit and a branch?",
        relevant_document_ids=["doc_0015"],
    ),
    EvalQuery(
        query="What HTTP methods are used in a RESTful API?",
        relevant_document_ids=["doc_0016"],
    ),
    EvalQuery(
        query="Why would you add an index to a database column?",
        relevant_document_ids=["doc_0017"],
    ),
]
