from openai import OpenAI
from zhipuai import ZhipuAI

from config import GLM_KEY


def zhipu_generate(text, model="glm-4", system=""):
    if model == '':
        model = "glm-4"
    client = ZhipuAI(api_key=GLM_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    response = client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        max_tokens=2048,
        messages=messages,
    )
    return response.choices[0].message.content


def openai_generate(text, model, system=""):
    client = OpenAI(
        # This is the default and can be omitted
        api_key="sk-X7Zbbb2bb1bdce35f10f991e70155dc66984ac21d1102tvb",
        base_url="https://api.gptsapi.net/v1",  # https://help.bewildcard.com/zh-CN/articles/9121334-wildcard-api-使用教程
    )
    if model == '':
        model = "gpt-3.5-turbo"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    reply = response.choices[0].message.content
    return reply
