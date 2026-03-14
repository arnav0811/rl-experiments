from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
# print(dataset[0]["problem"])
# print(dataset[0]["solution"])

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

problem = dataset[0]["problem"]
messages = [
    {"role": "system", "content": "Solve the following math problem. Think step by step, then give your final answer. You MUST put your final answer inside \\boxed{}."},
    {"role": "user", "content": problem},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize = True,
    return_dict = True,
    return_tensors = "pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

def extract_answer(model_response):
    idx = model_response.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(model_response) and depth > 0:
        if model_response[i] == '{':
            depth += 1
        elif model_response[i] == '}':
            depth -= 1
        i += 1
    
    if depth != 0:
        return None

    model_answer = model_response[start:i - 1]
    return model_answer

def check_answer(model_answer, ground_truth):
    model_answer = extract_answer(model_answer)
    if model_answer is None:
        return False
    return verify(parse(model_answer), parse(ground_truth))

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
# print(response)
print(dataset[0]["answer"])
print(extract_answer(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])))
print(check_answer(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]), dataset[0]["answer"]))