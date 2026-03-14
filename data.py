from datasets import load_dataset

def get_dataset():
    return load_dataset("HuggingFaceH4/MATH-500", split="test")

# dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
# print(dataset[0]["problem"])
# print(dataset[0]["solution"])