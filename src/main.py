import json
import os

from easyrag.pipeline.pipeline import EasyRAGPipeline
from submit import submit
import fire
from tqdm.asyncio import tqdm
from easyrag.pipeline.qa import read_jsonl, save_answers, write_jsonl
from easyrag.utils import get_yaml_data


def get_test_data(split="val"):
    if split == 'test':
        queries = read_jsonl("data/question.jsonl")
    else:
        with open("data/val.json") as f:
            queries = json.loads(f.read())
    return queries


async def main(
        re_only=False,
        split='test',  # 使用哪个集合
        push=False,  # 是否直接提交这次test结果
        save_inter=True,  # 是否保存检索结果等中间结果
        note="best",  # 中间结果保存路径的备注名字
        config_path="configs/easyrag.yaml",  # 配置文件
):
    # 读入配置文件
    config = get_yaml_data(config_path)
    config['re_only'] = re_only
    for key in config:
        print(f"{key}: {config[key]}")

    # 创建RAG流程
    rag_pipeline = EasyRAGPipeline(
        config
    )

    # 读入测试集
    queries = get_test_data(split)

    # 生成答案
    print("开始生成答案...")
    answers = []
    all_nodes = []
    all_contexts = []
    for query in tqdm(queries, total=len(queries)):
        res = await rag_pipeline.run(query)
        answers.append(res['answer'])
        all_nodes.append(res['nodes'])
        all_contexts.append(res['contexts'])

    # 处理结果
    print("处理生成内容...")
    os.makedirs("outputs", exist_ok=True)

    # 本地提交
    answer_file = f"outputs/submit_result_{split}_{note}.jsonl"
    answers = save_answers(queries, answers, answer_file)
    print(f"保存结果至 {answer_file}")

    # docker提交
    answer_file = f"submit_result.jsonl"
    write_jsonl(answer_file, answers)

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
        print("保存中间结果...")
        inter_res_list = []
        for query, answer, nodes, contexts in tqdm(zip(queries, answers, all_nodes, all_contexts)):
            paths = [node.metadata['file_path'] for node in nodes]
            know_paths = [node.metadata['know_path'] for node in nodes]
            inter_res = {
                "id": query['id'],
                "query": query['query'],
                "answer": answer['answer'],
                "candidates": contexts,
                "paths": paths,
                "know_paths": know_paths,
                "quality": [0 for _ in range(len(contexts))],
                "score": 0,
                "duplicate": 0,
            }
            if 'keywords' in query:
                inter_res['keywords'] = query['keywords']
                inter_res['gt'] = query['answer']
            inter_res_list.append(inter_res)
        inter_file = f"inter/{split}_{note}.json"
        with open(f"{inter_file}", 'w') as f:
            f.write(json.dumps(inter_res_list, ensure_ascii=False, indent=4))
        print(f"保存中间结果至 {inter_file}")


if __name__ == "__main__":
    fire.Fire(main)
