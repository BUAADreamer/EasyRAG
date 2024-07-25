from typing import List, Dict, Any

from llama_index.core.node_parser import HierarchicalNodeParser

from ..custom.transformation import CustomFilePathExtractor, CustomTitleExtractor
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from ..custom.splitter import SentenceSplitter
from ..custom.hierarchical import HierarchicalNodeParser
from llama_index.core.schema import Document, MetadataMode, TransformComponent
from llama_index.core.vector_stores.types import BasePydanticVectorStore, MetadataFilters, MetadataFilter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def get_node_content(node, embed_type=0) -> str:
    text = node.get_content()
    if embed_type == 1:
        text = '###\n' + node.metadata['file_path'] + "\n\n" + text
    elif embed_type == 2:
        text = '###\n' + node.metadata['know_path'] + "\n\n" + text
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
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        config: dict, reindex: bool = False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    client = AsyncQdrantClient(
        # url=config["QDRANT_URL"],
        # location=":memory:",
        path=config['cache_path'],
    )
    if reindex:
        try:
            await client.delete_collection(config["COLLECTION_NAME"])
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

    try:
        await client.create_collection(
            collection_name=config["COLLECTION_NAME"],
            vectors_config=models.VectorParams(
                size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.COSINE
            ),
        )
    except Exception as e:
        print("集合已存在")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=config["COLLECTION_NAME"],
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
