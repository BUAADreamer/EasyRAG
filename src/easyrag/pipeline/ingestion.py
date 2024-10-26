from typing import List, Dict, Any

from llama_index.core.node_parser import HierarchicalNodeParser

from ..custom.transformation import CustomFilePathExtractor, CustomTitleExtractor
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from ..custom.splitter import SentenceSplitter
from ..custom.hierarchical import HierarchicalNodeParser
from llama_index.core.schema import Document, MetadataMode, TransformComponent, NodeRelationship, TextNode, NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, MetadataFilters, MetadataFilter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def merge_strings(A, B):
    # 找到A的结尾和B的开头最长的匹配子串
    max_overlap = 0
    min_length = min(len(A), len(B))

    for i in range(1, min_length + 1):
        if A[-i:] == B[:i]:
            max_overlap = i

    # 合并A和B，去除重复部分
    merged_string = A + B[max_overlap:]
    return merged_string


def get_node_content(node: NodeWithScore, embed_type=0, nodes: list[TextNode] = None, nodeid2idx: dict = None) -> str:
    text: str = node.get_content()
    if embed_type == 6:
        cur_text = text
        if cur_text.count("|") >= 5 and cur_text.count("---") == 0:
            cnt = 0
            flag = False
            while True:
                pre_node_id = node.node.relationships[NodeRelationship.PREVIOUS].node_id
                pre_node = nodes[nodeid2idx[pre_node_id]]
                pre_text = pre_node.text
                cur_text = merge_strings(pre_text, cur_text)
                cnt += 1
                if pre_text.count("---") >= 2:
                    flag = True
                    break
                if cnt >= 3:
                    break
            if flag:
                idx = cur_text.index("---")
                text = cur_text[:idx].strip().split("\n")[-1] + cur_text[idx:]
            # print(flag, cnt)
    if embed_type == 1:
        if 'file_path' in node.metadata:
            text = '###\n' + node.metadata['file_path'] + "\n\n" + text
    elif embed_type == 2:
        if 'know_path' in node.metadata:
            text = '###\n' + node.metadata['know_path'] + "\n\n" + text
    elif embed_type == 3 or embed_type == 6:
        if "imgobjs" in node.metadata and len(node.metadata['imgobjs']) > 0:
            for imgobj in node.metadata['imgobjs']:
                text = text.replace(f"{imgobj['cap']} {imgobj['title']}\n", f"{imgobj['cap']}.{imgobj['title']}:{imgobj['content']}\n")
    elif embed_type == 4:
        if 'file_path' in node.metadata:
            text = node.metadata['file_path']
        else:
            text = ""
    elif embed_type == 5:
        if 'know_path' in node.metadata:
            text = node.metadata['know_path']
        else:
            text = ""
    return text


def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
    )
    return reader.load_data()


def build_preprocess(
        data_path=None,
        chunk_size=1024,
        chunk_overlap=50,
        split_type=0,  # 0-->Sentence 1-->Hierarchical
) -> List[TransformComponent]:
    if split_type == 0:
        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_prev_next_rel=True,
        )
    else:
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[chunk_size * 4, chunk_size],
            chunk_overlap=chunk_overlap,
        )
    transformation = [
        parser,
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=100000, data_path=data_path, metadata_mode=MetadataMode.EMBED),
    ]
    return transformation


def build_preprocess_pipeline(
        data_path=None,
        chunk_size=1024,
        chunk_overlap=50,
        split_type=0,
) -> IngestionPipeline:
    transformation = build_preprocess(
        data_path,
        chunk_size,
        chunk_overlap,
        split_type=split_type,
    )
    return IngestionPipeline(transformations=transformation)


def build_pipeline(
        llm: LLM,
        embed_model: BaseEmbedding,
        template: str = None,
        vector_store: BasePydanticVectorStore = None,
        data_path=None,
        chunk_size=1024,
        chunk_overlap=50,
) -> IngestionPipeline:
    transformation = build_preprocess(
        data_path,
        chunk_size,
        chunk_overlap,
    )
    transformation.extend([
        # SummaryExtractor(
        #     llm=llm,
        #     metadata_mode=MetadataMode.EMBED,
        #     prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        # ),
        embed_model,
    ])
    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


async def build_vector_store(
        qdrant_url: str = "http://localhost:6333",
        cache_path: str = "cache",
        reindex: bool = False,
        collection_name: str = "aiops24",
        vector_size: int = 3584,
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    if qdrant_url:
        client = AsyncQdrantClient(
            url=qdrant_url,
        )
    else:
        client = AsyncQdrantClient(
            path=cache_path,
        )

    if reindex:
        try:
            await client.delete_collection(collection_name)
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
    except Exception as e:
        print("集合已存在")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=collection_name,
        parallel=4,
        batch_size=32,
    )


def build_filters(dir):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="dir",
                # operator=FilterOperator.EQ,
                value=dir,
            ),
        ]
    )
    return filters


def build_qdrant_filters(dir):
    filters = Filter(
        must=[
            FieldCondition(
                key="dir",
                match=MatchValue(value=dir),
            )
        ]
    )
    return filters
