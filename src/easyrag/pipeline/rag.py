from llama_index.core.base.llms.types import CompletionResponse


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


def analysis_path_res(query, node_with_scores):
    num = len(node_with_scores)
    if num < 192 or query == 'VNF弹性分几类？':
        print(query, num)
    else:
        return
    # if len(node_with_scores) > 20 and query != 'VNF弹性分几类？':
    #     return
    pathmap = dict()
    for node in node_with_scores[:10]:
        know_path = node.metadata['file_path']
        print(know_path)
        if know_path not in pathmap:
            pathmap[know_path] = 1
        else:
            pathmap[know_path] += 1
    # print(query)
    # print(len(pathmap.keys()))
    # print(pathmap)
