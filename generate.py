from rewards import extract_answer, check_answer
from data import get_dataset
from model import load_model

def generate_and_eval(model, tokenizer, dataset, idx = 0):
    problem = dataset[idx]["problem"]
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

    print(dataset[idx]["answer"])
    print(extract_answer(response))
    print(check_answer(response, dataset[idx]["answer"]))
