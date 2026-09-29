"""PrePostPipelineEncoder: small embedding model + query/code preprocessing.

Change ONLY the settings block below between experiments, and bump RUN_ID
(mteb caches results by model name + revision, so a new RUN_ID forces a recompute).
"""
import re

from sentence_transformers import SentenceTransformer
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

# ---------------- settings (edit these) ----------------
RUN_ID = "3"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOC_PREFIX = ""
USE_CLEANING = False
MAX_QUERY_WORDS = 300
# --------------------------------------------------------


def split_identifiers(text: str) -> str:
    """maxSum / max_sum -> 'maxSum max sum' (keeps the original too)."""
    def _split(m):
        w = m.group(0)
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ")
        return w if parts == w else f"{w} {parts.lower()}"
    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _split, text)


def clean_query(text: str) -> str:
    """APPS problems: drop the sample I/O block, collapse whitespace, truncate."""
    text = re.split(r"-{3,}\s*Examples?\s*-{3,}", text)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:MAX_QUERY_WORDS])


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
            name=f"local/prepost-pipeline-encoder-run{RUN_ID}",
            revision=RUN_ID,
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
            if USE_CLEANING:
                texts = [clean_query(t) for t in texts]
            texts = [QUERY_PREFIX + t for t in texts]
        else:
            if USE_CLEANING:
                texts = [clean_code(t) for t in texts]
            texts = [DOC_PREFIX + t for t in texts]
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=True,
        )
