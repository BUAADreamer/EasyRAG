import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


def convert_transparent_to_white(image_path):
    # 打开图像
    img = Image.open(image_path).convert("RGBA")

    # 创建一个白色背景图像
    white_background = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # 复合图像，将透明部分用白色填充
    composite = Image.alpha_composite(white_background, img)

    # 转换为RGB模式，去除透明度
    final_img = composite.convert("RGB")

    return final_img


def glm4v_generate(query="简要描述图像", img_path="temp/1.png"):
    image = convert_transparent_to_white(img_path)
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True)  # chat mode

    inputs = inputs.to(device)
    # gen_kwargs = {"max_length": 4096, "temperature": 0.1, "do_sample": True, "top_k": 1, "top_p": 1, "repetition_penalty": 1}
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        res = tokenizer.decode(outputs[0])
        res = res.replace("<|endoftext|>", "")
        res = res.strip()
    return res
