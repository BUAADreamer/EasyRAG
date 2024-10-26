import re

from llama_index.core.base.llms.types import CompletionResponse


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


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
                print(f"已达到最大生成次数{cnt}次，返回'无法确定'")
                return CompletionResponse(text="无法确定")
            print(f"已重复生成{cnt}次")


def deduplicate(contents):
    new_contents = []
    contentmap = dict()
    for content in contents:
        if content not in contentmap:
            contentmap[content] = 1
            new_contents.append(content)
    return new_contents
