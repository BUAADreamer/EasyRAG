import torch
from zhipuai import ZhipuAI
import base64

client = ZhipuAI(api_key="4f3c198f8bcbfa81e2ef9fbea428d9d8.DGBE51vKXQP65ceE")


def zhipu_generate(text, model="glm-4", system=""):
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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def zhipu_generate_vision(image_path, prompt, model="glm-4v", system=""):
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
            }
        },
    ]
    reply = zhipu_chat_vision(content, model, system)
    return reply


# vision:https://platform.openai.com/docs/guides/vision
def zhipu_chat_vision(content: list, model="glm-4v", system=""):
    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": content})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        top_p=0.7,
        temperature=0.95,
        max_tokens=1024,
    )
    print(response)
    reply = response.choices[0].message.content
    return reply


def local_llm_generate(query, model, tokenizer, device="cuda"):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 8192, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res
