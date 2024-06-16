from typing import List
import qdrant_client

from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse

from custom.template import QA_TEMPLATE



async def generation_with_knowledge_retrieval(
        query_str: str,
        retriever: BaseRetriever,
        llm: LLM,
        qa_template: str = QA_TEMPLATE,
        reranker: BaseNodePostprocessor = None,
        debug: bool = False,
        progress=None,
        re_only: bool = False,
):
    query_bundle = QueryBundle(query_str=query_str)
    node_with_scores = await retriever.aretrieve(query_bundle)
    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    if re_only:
        return CompletionResponse(text=""), node_with_scores
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret, node_with_scores
