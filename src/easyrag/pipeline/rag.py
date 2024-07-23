import time

from ..custom.template import QA_TEMPLATE
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
)
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from ..custom.retrievers import HybridRetriever


def filter_specfic_words(prompt):
    word_dict = {
        "支持\nZDB": "ZDB"
    }
    for key, value in word_dict.items():
        prompt = prompt.replace(key, value)
    return prompt


async def generation(llm, fmt_qa_prompt):
    cnt = 0
    # fmt_qa_prompt = filter_specfic_words(fmt_qa_prompt)
    while True:
        try:
            ret = await llm.acomplete(fmt_qa_prompt)
            return ret
        except Exception as e:
            print(e)
            cnt += 1
            if cnt >= 10:
                return CompletionResponse(text="无法确定")
            print(f"已重复生成{cnt}次")


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
        [f"### 文档{i}: {node.text}" for i, node in enumerate(node_with_scores)]
    )

    if re_only:
        return CompletionResponse(text=""), node_with_scores
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await generation(llm, fmt_qa_prompt)
    return ret, node_with_scores


async def generation_with_rerank_fusion(
        query_str: str,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        llm: LLM,
        qa_template: str = QA_TEMPLATE,
        reranker: BaseNodePostprocessor = None,
        progress=None,
        re_only: bool = False,
        rerank_fusion_type=1,
        r_topk_1=6,
):
    query_bundle = QueryBundle(query_str=query_str)

    node_with_scores_dense = await dense_retriever.aretrieve(query_bundle)
    if reranker:
        node_with_scores_dense = reranker.postprocess_nodes(node_with_scores_dense, query_bundle)

    node_with_scores_sparse = await sparse_retriever.aretrieve(query_bundle)
    if reranker:
        node_with_scores_sparse = reranker.postprocess_nodes(node_with_scores_sparse, query_bundle)

    node_with_scores = HybridRetriever.reciprocal_rank_fusion([node_with_scores_sparse, node_with_scores_dense],
                                                              topk=r_topk_1)
    # node_with_scores = HybridRetriever.fusion([node_with_scores_sparse, node_with_scores_dense], topk=reranker.top_n)

    if re_only:
        return CompletionResponse(text=""), node_with_scores

    if rerank_fusion_type == 1:
        context_str = "\n\n".join(
            [f"### 文档{i}: {node.text}" for i, node in enumerate(node_with_scores_dense)]
        )
        fmt_qa_prompt = PromptTemplate(qa_template).format(
            context_str=context_str, query_str=query_str
        )
        ret = await generation(llm, fmt_qa_prompt)
    else:
        context_str = "\n\n".join(
            [f"### 文档{i}: {node.text}" for i, node in enumerate(node_with_scores_sparse)]
        )
        fmt_qa_prompt = PromptTemplate(qa_template).format(
            context_str=context_str, query_str=query_str
        )
        ret_sparse = await generation(llm, fmt_qa_prompt)

        context_str = "\n\n".join(
            [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores_dense]
        )
        fmt_qa_prompt = PromptTemplate(qa_template).format(
            context_str=context_str, query_str=query_str
        )
        ret_dense = await generation(llm, fmt_qa_prompt)

        if rerank_fusion_type == 2:
            if len(ret_dense.text) >= len(ret_sparse.text):
                ret = ret_dense
            else:
                ret = ret_sparse
        elif rerank_fusion_type == 3:
            ret = ret_sparse + ret_dense

    return ret, node_with_scores
