import json
import os
import random

import fire
from tqdm import tqdm

from pipeline.ingestion import read_data
from pipeline.llm_utils import zhipu_generate, openai_generate
from dotenv import dotenv_values


def get_json(text):
    text = text.replace("```json", "")
    text = text.replace("```", "")
    idx = text.index("}")
    text = text[:idx + 1]
    return json.loads(text)


def gen_val(num, out):
    config = dotenv_values(".env")
    data_path = config.get("data_path", "data")
    documents = read_data(data_path)
    print(documents[0].metadata)
    return
    print(f"文档加载成功，共有{len(documents)}个文档")
    prompt_template = """# 上下文
你是一个通信网络运维专家，负责根据中兴通讯公司CT通信网络运维下真实文档数据，标注问答对

# 目标
分析利用给定文档能解决的最相关问题，以及给出问题对应的答案

请按照以下步骤思考：
1.先生成问题
2.再生成问题对应的答案
3.再根据答案，提取出和运维相关的至少5个重要技术关键词

最后以json格式返回

# 响应
返回一个json对象，其中query字段代表问题，answer字段代表答案，keywords字段代表关键词列表
不要解释生成的内容，不要返回任何其他内容

# 文档
{0}

# 生成的json对象
"""
    selected_documents = random.sample(documents, num)
    val_data_ls = []
    for doc in tqdm(selected_documents):  # doc: text/doc_id/extra_info
        prompt = prompt_template.format(doc.text)
        while True:
            res = zhipu_generate(prompt)
            try:
                data = get_json(res)
            except:
                print(res)
                continue
            val_data_ls.append(data)
            break
    with open(out, 'w') as f:
        f.write(json.dumps(val_data_ls, ensure_ascii=False, indent=4))


def main(
        split="val",  # val train
        seed=42,
        num=1,
        out="dataset/val.json",
):
    random.seed(seed)
    if split == 'val':
        gen_val(num, out)
    elif split == 'train':
        pass


if __name__ == '__main__':
    fire.Fire(main)
