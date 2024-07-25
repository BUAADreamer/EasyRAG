import json

from paddleocr import PaddleOCR, draw_ocr
import re
from easyrag.utils.llm_utils import zhipu_generate_vision

json_path = "/home/zhangrichong/data/fengzc/rag/wzy_rag/RAG-AIOps/data/format_data_with_img/imgmap.json"


def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]+', s))


def get_content(img_path="1.png"):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    content = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            content += line + "\n"
    return content


def get_cnt(text):
    pattern = r'(?<!流程)如图\d+所示'
    matches = re.findall(pattern, text)
    print("匹配项：", matches)
    return len(matches)


def match_content(text, cap):
    s = '(?<!流程)如{}所示'.format(cap)
    pattern = re.escape(s)
    matches = re.findall(pattern, text)
    return len(matches)


# with open(json_path) as f:
#     img_map = json.loads(f.read())
#     cnt = 0
#     for file_path in img_map:
#         cnt += len(img_map[file_path].keys())
#     print(cnt)

if __name__ == "__main__":
    img_path = "temp/1.png"
    prompt = """### 目标
    
    你是一个理解图像的小助手，总结图中内容
    
    ### 要求
    
    需要出现名词时都使用图中的原词，不要修改
    
    如果有英文名词，不要修改
    
    ### 返回格式
    
    直接返回总结的文本内容
    """
    prompt = "简要描述图像"
    res = zhipu_generate_vision(img_path, prompt)
    print(res)
