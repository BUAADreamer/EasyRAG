import asyncio
import json
import os
from submit import submit
import fire
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data, build_filters
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from config import GLM_KEY


def get_test_data(split="val"):
    if split == 'test':
        queries = read_jsonl("question.jsonl")[:5]
    else:
        with open("dataset/val.json") as f:
            queries = json.loads(f.read())
    return queries


async def main(
        split='val',
        push=False,
        save_inter=True,
        note="",
):
    config = dotenv_values(".env")

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = OpenAI(
        api_key=GLM_KEY,
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    embedding = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-zh-v1.5",
        cache_folder=config.get("hfmodel_cache_folder"),
        embed_batch_size=128,
        # query_instruction="为这个句子生成表示以用于检索相关文章：", # 默认已经加上了，所以加不加无所谓
    )
    Settings.embed_model = embedding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )
    data_path = config.get("data_path", "data")
    if collection_info.points_count == 0:
        data = read_data(data_path)
        pipeline = build_pipeline(llm, embedding, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))

    # 读入数据集
    queries = get_test_data(split)

    # 生成答案
    print("Start generating answers...")

    results = []
    docs = []
    for query in tqdm(queries, total=len(queries)):
        if "document" in query:
            filters = build_filters(
                dir=os.path.join(data_path, query['document'])
            )
        else:
            filters = None
        retriever = QdrantRetriever(vector_store, embedding, similarity_top_k=8, filters=filters)
        result, contexts = await generation_with_knowledge_retrieval(
            query["query"], retriever, llm
        )
        docs.append(contexts)
        results.append(result)

    # 处理结果
    os.makedirs("outputs", exist_ok=True)
    answers = save_answers(queries, results, f"outputs/submit_result_{split}.jsonl")

    # 做评测
    os.makedirs("inter", exist_ok=True)
    N = len(queries)
    if split == 'test':
        if push:
            judge_res = submit(answers)
            print(judge_res)

    elif split == 'val':
        all_keyword_acc = 0
        all_sim_acc = 0
        for answer_obj, gt_obj in tqdm(zip(answers, queries)):
            answer = answer_obj['answer']
            keywords = gt_obj['keywords']
            gt = gt_obj['answer']
            M = len(keywords)
            keyword_acc = 0
            for keyword in keywords:
                if keyword in answer:
                    keyword_acc += 1
            keyword_acc /= M
            all_keyword_acc += keyword_acc
        all_keyword_acc /= N
        all_sim_acc /= N
        acc = all_keyword_acc
        print("average acc:", acc * 100)

    # 保存中间结果
    if save_inter:
        inter_res_list = []
        for query, answer, documents in tqdm(zip(queries, answers, docs)):
            contexts = [f"{doc.metadata['document_title']}: {doc.text}" for doc in documents]
            paths = [doc.metadata['file_path'] for doc in documents]
            inter_res = {
                "id": query['id'],
                "query": query['query'],
                "answer": answer['answer'],
                "candidates": contexts,
                "paths": paths,
                "quality": [0 for _ in range(len(contexts))],
                "score": 0,
                "duplicate": 0,
            }
            if 'keywords' in query:
                inter_res['keywords'] = query['keywords']
                inter_res['gt'] = query['answer']
            inter_res_list.append(inter_res)
        with open(f"inter/{split}_{note}.json", 'w') as f:
            f.write(json.dumps(inter_res_list, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    fire.Fire(main)
