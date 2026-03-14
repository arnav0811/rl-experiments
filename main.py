from rewards import extract_answer, check_answer
from model import load_model
from data import get_dataset

dataset = get_dataset()
model, tokenizer = load_model()

# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

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

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])


# print(response)
print(dataset[0]["answer"])
print(extract_answer(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])))
print(check_answer(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]), dataset[0]["answer"]))