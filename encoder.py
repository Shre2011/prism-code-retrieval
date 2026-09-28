"""PrePostPipelineEncoder: small embedding model + query/code preprocessing.

NOTE: written against the mteb v2 API shown in the hackathon guideline, but not
run here. If your installed mteb version complains about a signature or a
missing ModelMeta field, follow the error message (or print
inspect.signature(AbsEncoder.encode)) and adjust.
"""
import re

from sentence_transformers import SentenceTransformer
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # small, fast on CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
USE_PREFIX = True          # toggle and compare scores
MAX_QUERY_WORDS = 300


def split_identifiers(text: str) -> str:
    """maxSum / max_sum -> 'max sum' (keeps the original too)."""
    def _split(m):
        w = m.group(0)
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ")
        return w if parts == w else f"{w} {parts.lower()}"
    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _split, text)


def clean_query(text: str) -> str:
    """APPS problems: drop the sample I/O block, collapse whitespace, truncate."""
    text = re.split(r"-{3,}\s*Examples?\s*-{3,}", text)[0]
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:MAX_QUERY_WORDS])


def clean_code(text: str) -> str:
    """Strip '#' comments, normalise whitespace, split identifiers."""
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return split_identifiers(text)


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.model.max_seq_length = 512
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name="local/prepost-pipeline-encoder",
            revision="1",
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=512,
            embed_dim=384,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["Sentence Transformers"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        if prompt_type == PromptType.query:
            texts = [clean_query(t) for t in texts]
            if USE_PREFIX:
                texts = [QUERY_PREFIX + t for t in texts]
        else:
            texts = [clean_code(t) for t in texts]
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=True,
        )
