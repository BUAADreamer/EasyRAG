import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from ...pipeline.ingestion import get_node_content
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import BaseNode
from llama_index.core.utils import get_cache_dir, infer_torch_device
from llama_index.embeddings.huggingface.utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)
from sentence_transformers import SentenceTransformer

DEFAULT_HUGGINGFACE_LENGTH = 512
logger = logging.getLogger(__name__)


class HuggingFaceEmbedding(BaseEmbedding):
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for Hugging Face files."
    )

    _model: Any = PrivateAttr()
    _device: str = PrivateAttr()
    _embed_type: int = PrivateAttr()

    def __init__(
            self,
            model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
            tokenizer_name: Optional[str] = "deprecated",
            pooling: str = "deprecated",
            max_length: Optional[int] = None,
            query_instruction: Optional[str] = None,
            text_instruction: Optional[str] = None,
            normalize: bool = True,
            model: Optional[Any] = "deprecated",
            tokenizer: Optional[Any] = "deprecated",
            embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
            cache_folder: Optional[str] = None,
            trust_remote_code: bool = False,
            device: Optional[str] = None,
            callback_manager: Optional[CallbackManager] = None,
            embed_type: int = 0,
            **model_kwargs,
    ):
        self._device = device or infer_torch_device()
        self._embed_type = embed_type
        cache_folder = cache_folder or get_cache_dir()

        for variable, value in [
            ("model", model),
            ("tokenizer", tokenizer),
            ("pooling", pooling),
            ("tokenizer_name", tokenizer_name),
        ]:
            if value != "deprecated":
                raise ValueError(
                    f"{variable} is deprecated. Please remove it from the arguments."
                )
        if model_name is None:
            raise ValueError("The `model_name` argument must be provided.")

        self._model = SentenceTransformer(
            model_name,
            device=self._device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            prompts={
                "query": query_instruction
                         or get_query_instruct_for_model_name(model_name),
                "text": text_instruction
                        or get_text_instruct_for_model_name(model_name),
            },
            **model_kwargs,
        )
        if max_length:
            self._model.max_seq_length = max_length
        else:
            max_length = self._model.max_seq_length

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"

    def _embed(
            self,
            sentences: List[str],
            prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed sentences."""
        return self._model.encode(
            sentences,
            batch_size=self.embed_batch_size,
            prompt_name=prompt_name,
            normalize_embeddings=self.normalize,
        ).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed(query, prompt_name="query")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed(text, prompt_name="text")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, prompt_name="text")

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        embeddings = self.get_text_embedding_batch(
            [get_node_content(node, self._embed_type) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    async def acall(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        embeddings = await self.aget_text_embedding_batch(
            [get_node_content(node, self._embed_type) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes
