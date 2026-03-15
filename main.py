from model import load_model
from data import get_dataset
from train import train_sft, train_grpo
import os

def main():
    dataset = get_dataset()
    model, tokenizer = load_model()
    if os.path.exists("checkpoints/sft"):
        model = PeftModel.from_pretrained("checkpoints/sft")
        print("Loaded SFT model")
    else:
        model = train_sft(model, tokenizer, dataset)
    model = train_grpo(model, tokenizer, dataset)

if __name__ == "__main__":
    main()