from model import load_model
from data import get_dataset
from train import train

def main():
    dataset = get_dataset()
    model, tokenizer = load_model()
    model = train(model, tokenizer, dataset)

if __name__ == "__main__":
    main()