import json
import os.path

import fire
from paddleocr import PaddleOCR
import re

from tqdm import tqdm

from easyrag.utils.mllm_utils import glm4v_generate


def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]+', s))


ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory


def get_ocr_content(img_path="1.png"):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    result = ocr.ocr(img_path, cls=True)
    content = ""
    for idx in range(len(result)):
        try:
            res = result[idx]
            for line in res:
                content += line[1][0] + "\n"
        except:
            continue
    return content


def get_cnt(text):
    pattern = r'(?<!流程)如图\d+所示'
    matches = re.findall(pattern, text)
    print("匹配项：", matches)
    return len(matches)


def match_content(text, cap):
    ignore_list = [
        "流程", "，", "示例", "配置", "组网图", "（可选）", "文件"
    ]
    s = '(?<!{ignore_str})如{cap}所示'.format(ignore_str="|".join(ignore_list), cap=cap)
    pattern = re.escape(s)
    matches = re.findall(pattern, text)
    return len(matches)


def get_image_num(img_map):
    cnt = 0
    for file_path in img_map:
        cnt += len(img_map[file_path].keys())
    return cnt


def filter(title):
    ignore_words = ["架构", "结构", "组网图", "页面", "对话框"]
    for ignore_word in ignore_words:
        if ignore_word in title:
            return True
    return False


def main(

):
    json_path = "../data/format_data_with_img/imgmap_raw.json"
    image_root = "../data/format_data_with_img"
    with open(json_path) as f:
        img_map = json.loads(f.read())
        all_cnt = get_image_num(img_map)
        print("原始图像个数", all_cnt)
    json_output_path = "../data/format_data_with_img/imgmap_filtered.json"
    new_img_map = dict()
    post_cnt = 0
    cnt = 0
    w = tqdm(img_map)
    for file_path in w:
        for img_name in img_map[file_path].keys():
            img_path = img_map[file_path][img_name]["img_path"]
            img_abs_path = os.path.join(image_root, img_path)
            img_title = img_map[file_path][img_name]["title"]
            flag = False
            if "content" in img_map[file_path][img_name]:
                flag = True
            else:
                ocr_content = get_ocr_content(img_abs_path)
                if ocr_content != "" and contains_chinese(ocr_content):
                    flag = True
            if flag:
                post_cnt += 1
                if file_path not in new_img_map:
                    new_img_map[file_path] = {}
                new_img_map[file_path][img_name] = img_map[file_path][img_name]
                try:
                    content = glm4v_generate(img_path=img_abs_path)
                except:
                    continue
                new_img_map[file_path][img_name]['content'] = content
            cnt += 1
            w.set_description(f"已处理{cnt}个图像,已获取{post_cnt}个图像")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_img_map, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    fire.Fire(main)
