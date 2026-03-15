from peft import get_peft_model, LoraConfig, TaskType
from data import get_dataloader
from torch.optim import AdamW

def train(model, tokenizer, dataset, num_epochs = 3, lr = 2e-4):
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        # rank
        r = 16,
        # scaling factor
        lora_alpha = 32,
        lora_dropout = 0.1,
        target_modules = ["q_proj", "v_proj"] 
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    dataloader = get_dataloader(dataset, tokenizer)
    optimizer = AdamW(model.parameters(), lr = lr)

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            # forward pass
            # shifts labels left by 1 position to create next token prediction targets
            # runs cross entropy loss etween the model's logits and those shifted targets
            # ignores positions where label = -100 (padding tokens)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
    
    model.save_pretrained("checkpoints/sft")
    tokenizer.save_pretrained("checkpoints/sft")
    return model