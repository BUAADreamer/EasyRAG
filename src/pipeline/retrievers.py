from typing import List

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore


class QdrantRetriever(BaseRetriever):
    def __init__(
            self,
            vector_store: QdrantVectorStore,
            embed_model: BaseEmbedding,
            similarity_top_k: int = 2,
            filters=None
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        self.filters = filters
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding,
            similarity_top_k=self._similarity_top_k,
            # filters=self.filters, # qdrant 使用llama_index filter会有问题，原因未知
        )
        query_result = await self._vector_store.aquery(
            vector_store_query,
            qdrant_filters=self.filters,  # 需要查找qdrant相关用法
        )

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 不维护
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding,
            similarity_top_k=self._similarity_top_k,
        )
        query_result = self._vector_store.query(
            vector_store_query,
            qdrant_filters=self.filters,  # 需要查找qdrant相关用法
        )

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


class HybridRetriever(BaseRetriever):
    def __init__(
            self,
            dense_retriever: QdrantRetriever,
            sparse_retriever: BM25Retriever,
            retrieval_type=3
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.retrieval_type = retrieval_type
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self.retrieval_type != 1:
            sparse_nodes = await self.sparse_retriever.aretrieve(query_bundle)
            if self.retrieval_type == 2:
                return sparse_nodes
        if self.retrieval_type != 2:
            dense_nodes = await self.dense_retriever.aretrieve(query_bundle)
            if self.retrieval_type == 1:
                return dense_nodes

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in sparse_nodes + dense_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 不维护 占位
        sparse_nodes = self.sparse_retriever.retrieve(query_bundle)
        dense_nodes = self.dense_retriever.retrieve(query_bundle)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in sparse_nodes + dense_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
