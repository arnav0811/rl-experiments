from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from functools import partial

def get_dataset():
    return load_dataset("HuggingFaceH4/MATH-500", split="test")

# dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
# print(dataset[0]["problem"])
# print(dataset[0]["solution"])

def format_example(example, tokenizer):
    messages = [
    {"role": "system", "content": "Solve the following math problem. Think step by step, then give your final answer. You MUST put your final answer inside \\boxed{}."},
    {"role": "user", "content": example["problem"]},
    {"role": "assistant", "content": example["solution"]},
    ]   

    tokenized = tokenizer.apply_chat_template(
        messages,
        # in SFT we are including the assistant response in the messages already
        add_generation_prompt = False,
        tokenize = True,
        return_dict = True,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # label s- prompt tokens are replaced with -100
    # we tokenize just the prompt without the assistant response to get its length
    prompt_messages = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        # True as we want the assistant header
        add_generation_prompt = True,
        tokenize = True,
    )
    prompt_len = len(prompt_ids)

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_dataloader(dataset, tokenizer, batch_size = 4):
    formatted_dataset = dataset.map(partial(format_example, tokenizer=tokenizer), num_proc=4)
    formatted_dataset = formatted_dataset.remove_columns(
        [col for col in formatted_dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )
    formatted_dataset.set_format("torch")
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)  
    dataloader = DataLoader(formatted_dataset, batch_size=batch_size, collate_fn=collator)
    return dataloader